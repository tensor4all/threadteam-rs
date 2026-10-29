/* The libgomp reference for benches/wait_scenarios.rs (docs/design/waiting.md).
 *
 * The same work as the Rust bench: a section is one `omp parallel` region in which every thread
 * busy-waits `work` microseconds on the clock; a call is a run of sections. Prints rows in the
 * Rust bench's format and appends them to a CSV file when one is given.
 *
 * Build and run with scripts/bench-omp.sh, or:
 *   gcc -O2 -fopenmp benches/omp/wait_omp.c -o target/wait_omp
 *   target/wait_omp <width> <calls> [csv] [label]
 * The OpenMP runtime runs with its defaults (no OMP_WAIT_POLICY, no GOMP_SPINCOUNT, no binding)
 * unless the caller sets them.
 */
#define _GNU_SOURCE
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static inline void relax(void) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

static void busy(double us) {
  double end = now_us() + us;
  while (now_us() < end) relax();
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static FILE *csv;
static char prefix[512];

static void row(const char *scenario, int width, const char *param, const char *metric, double *v,
                int n, double ideal) {
  qsort(v, n, sizeof *v, cmp);
  double med = v[n / 2], p90 = v[(n * 9 / 10) < n ? n * 9 / 10 : n - 1], max = v[n - 1];
  printf("%-9s %-10s %5d %-22s %-7s %11.1f %11.1f %11.1f %10.1f\n", scenario, "libgomp", width,
         param, metric, med, p90, max, med - ideal);
  if (csv)
    fprintf(csv, "%s,%s,libgomp,%d,%s,%s,%.2f,%.2f,%.2f,%.2f\n", prefix, scenario, width, param,
            metric, med, p90, max, med - ideal);
}

/* One call: the time of the first section and of the whole call, in us. */
static void call(int width, int sections, double work, double *first, double *whole) {
  double t0 = now_us();
  for (int s = 0; s < sections; s++) {
#pragma omp parallel num_threads(width)
    busy(work);
    if (s == 0) *first = now_us() - t0;
  }
  *whole = now_us() - t0;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: wait_omp <width> <calls> [csv] [label]\n");
    return 2;
  }
  int width = atoi(argv[1]), calls = atoi(argv[2]);
  cpu_set_t set;
  int cpus = sched_getaffinity(0, sizeof set, &set) == 0 ? CPU_COUNT(&set) : 0;
  char host[256] = "";
  gethostname(host, sizeof host - 1);
  const char *label = argc > 4 ? argv[4] : "";
  const char *load = getenv("WAIT_BENCH_LOAD") ? getenv("WAIT_BENCH_LOAD") : "none";
  snprintf(prefix, sizeof prefix, "%s,%s,%d,%s", host, label, cpus, load);
  if (argc > 3) {
    int fresh = access(argv[3], F_OK) != 0;
    csv = fopen(argv[3], "a");
    if (csv && fresh)
      fprintf(csv, "host,label,cpus,load,scenario,impl,width,param,metric,median_us,p90_us,max_us,overhead_us\n");
  }
  double *a = malloc(sizeof(double) * calls), *b = malloc(sizeof(double) * calls);
  char param[64];
  /* warm the hot team */
  for (int i = 0; i < 10; i++) call(width, 10, 5, a, b);

  int works[3] = {5, 20, 50};
  for (int k = 0; k < 3; k++) {
    for (int i = 0; i < calls; i++) call(width, 200, works[k], &a[i], &b[i]);
    snprintf(param, sizeof param, "work=%dus,sections=200", works[k]);
    row("sections", width, param, "call", b, calls, 200.0 * works[k]);
  }
  int gaps[4] = {0, 100, 1000, 10000};
  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < calls / 2; i++) {
      if (gaps[k]) usleep(gaps[k]);
      call(width, 10, 5, &a[i], &b[i]);
    }
    snprintf(param, sizeof param, "gap=%dus", gaps[k]);
    row("gaps", width, param, "first", a, calls / 2, 5);
    row("gaps", width, param, "call", b, calls / 2, 50);
  }
  if (csv) fclose(csv);
  return 0;
}
