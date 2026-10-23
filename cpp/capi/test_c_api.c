/* A plain C test of the coxdev C API: compiles as C99 against coxdev_c_api.h, checks
   basic results, error handling, and many create/free cycles. */
#include "coxdev_c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s (%s)\n", msg, coxdev_last_error()); failures++; } } while (0)

int main(void) {
    enum { N = 8 };
    double event[N] = {1, 2, 2, 3, 4, 4, 5, 6};
    int status[N] = {1, 1, 1, 0, 1, 1, 0, 1};
    double start[N] = {0, 0, 0.5, 0, 1, 0, 2, 0};
    int strata[N] = {0, 0, 1, 1, 0, 1, 0, 1};
    double eta[N] = {0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.2, 0.1};
    double grad[N], diag[N], hv[N], dev = 0, sat = 0;
    coxdev_t *h = NULL;
    int rc;

    CHECK(coxdev_api_version() == COXDEV_C_API_VERSION, "api version");

    /* basic evaluation, Efron, with start times and strata */
    rc = coxdev_new(N, start, event, status, strata, NULL, 1, &h);
    CHECK(rc == COXDEV_OK && h != NULL, "coxdev_new");
    CHECK(coxdev_n(h) == N, "coxdev_n");
    rc = coxdev_deviance(h, eta, NULL, &dev, grad, diag, &sat);
    CHECK(rc == COXDEV_OK && isfinite(dev) && dev > 0, "coxdev_deviance");

    /* the information times the all-ones vector is 0 within each stratum's
       risk sets only in special cases; just check it runs and is finite */
    {
        double v[N];
        int i;
        for (i = 0; i < N; i++) v[i] = 1.0;
        rc = coxdev_hessian_matvec(h, eta, NULL, v, hv);
        CHECK(rc == COXDEV_OK && isfinite(hv[0]), "coxdev_hessian_matvec");
    }

    /* the derivative of the information: symmetric in the direction and the
       argument, and 0 in the direction of all ones (a shift of eta within
       each stratum leaves the information unchanged) */
    {
        double u[N], v[N], ones[N], a[N], b[N], z[N];
        int i, ok = 1;
        for (i = 0; i < N; i++) { u[i] = sin(i + 1.0); v[i] = cos(2.0 * i); ones[i] = 1.0; }
        rc = coxdev_information_derivative_matvec(h, eta, NULL, v, u, a);
        CHECK(rc == COXDEV_OK, "coxdev_information_derivative_matvec");
        rc = coxdev_information_derivative_matvec(h, eta, NULL, u, v, b);
        CHECK(rc == COXDEV_OK, "coxdev_information_derivative_matvec (swapped)");
        rc = coxdev_information_derivative_matvec(h, eta, NULL, ones, u, z);
        CHECK(rc == COXDEV_OK, "coxdev_information_derivative_matvec (ones)");
        for (i = 0; i < N; i++) {
            if (fabs(a[i] - b[i]) > 1e-12 || fabs(z[i]) > 1e-12) ok = 0;
        }
        CHECK(ok, "information derivative: symmetric, and 0 in the direction of ones");
        CHECK(coxdev_information_derivative_matvec(NULL, eta, NULL, v, u, a) == COXDEV_ERR_ARGUMENT,
              "information derivative: NULL handle");
    }

    /* baseline hazard: size query then fill */
    {
        size_t m = 0;
        rc = coxdev_baseline_hazard(h, eta, NULL, &m, NULL, NULL, NULL, NULL);
        CHECK(rc == COXDEV_OK && m > 0, "baseline hazard size");
        {
            double *t = malloc(m * sizeof(double)), *hz = malloc(m * sizeof(double)), *ch = malloc(m * sizeof(double));
            int *st = malloc(m * sizeof(int));
            size_t m2 = m;
            rc = coxdev_baseline_hazard(h, eta, NULL, &m2, t, hz, ch, st);
            CHECK(rc == COXDEV_OK && m2 == m, "baseline hazard fill");
            free(t); free(hz); free(ch); free(st);
        }
    }

    /* sample_score: same seed, same draws */
    {
        double a[N * 3], b[N * 3];
        coxdev_t *h2 = NULL;
        rc = coxdev_new(N, NULL, event, status, NULL, NULL, 0, &h2);
        CHECK(rc == COXDEV_OK, "coxdev_new (no start, no strata)");
        rc = coxdev_sample_score(h2, eta, NULL, "multinomial", 3, 12345u, a);
        CHECK(rc == COXDEV_OK, "sample_score");
        rc = coxdev_sample_score(h2, eta, NULL, "multinomial", 3, 12345u, b);
        CHECK(rc == COXDEV_OK && memcmp(a, b, sizeof a) == 0, "sample_score reproducible");
        coxdev_free(h2);
    }

    /* error handling */
    rc = coxdev_deviance(NULL, eta, NULL, &dev, NULL, NULL, NULL);
    CHECK(rc == COXDEV_ERR_ARGUMENT && strlen(coxdev_last_error()) > 0, "NULL handle is an argument error");
    {
        int bad_status[N] = {1, 2, 1, 0, 1, 1, 0, 1};
        coxdev_t *hb = NULL;
        rc = coxdev_new(N, NULL, event, bad_status, NULL, NULL, 1, &hb);
        CHECK(rc == COXDEV_ERR_ARGUMENT && hb == NULL, "status 2 rejected");
        printf("bad status message: %s\n", coxdev_last_error());
    }
    rc = coxdev_sample_score(h, eta, NULL, "no-such-method", 2, 1u, hv);
    CHECK(rc != COXDEV_OK && strlen(coxdev_last_error()) > 0, "unknown sample_score method");
    printf("bad method message: %s\n", coxdev_last_error());
    coxdev_free(h);
    rc = coxdev_new(0, NULL, event, status, NULL, NULL, 1, &h);
    CHECK(rc == COXDEV_ERR_ARGUMENT, "n = 0 rejected");
    rc = coxdev_deviance(h, eta, NULL, &dev, NULL, NULL, NULL);  /* h is NULL now */
    CHECK(rc == COXDEV_ERR_ARGUMENT, "failed coxdev_new leaves *out NULL");

    /* many create/free cycles, with evaluations */
    {
        int k;
        for (k = 0; k < 20000; k++) {
            coxdev_t *hk = NULL;
            if (coxdev_new(N, start, event, status, (k % 2) ? strata : NULL, NULL, k % 3 == 0, &hk) != COXDEV_OK) { failures++; break; }
            if (coxdev_deviance(hk, eta, NULL, &dev, grad, diag, &sat) != COXDEV_OK) { failures++; }
            coxdev_free(hk);
        }
    }
    coxdev_free(NULL);

    if (failures == 0) printf("test_c_api: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
