/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2006 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written by Intel Corporation.
 *  Copyright (C) 2011-2017 Intel Corporation.  Intel provides this material
 *  to Argonne National Laboratory subject to Software Grant and Corporate
 *  Contributor License Agreement dated February 8, 2012.
 */

/* Header protection (i.e., IALLGATHERV_TSP_RING_ALGOS_H_INCLUDED) is
 * intentionally omitted since this header might get included multiple
 * times within the same .c file. */

#include "algo_common.h"
#include "tsp_namespace_def.h"

/* Routine to schedule a recursive exchange based allgatherv */
#undef FUNCNAME
#define FUNCNAME MPIR_TSP_Iallgatherv_sched_intra_ring
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_TSP_Iallgatherv_sched_intra_ring(const void *sendbuf, int sendcount,
                                          MPI_Datatype sendtype, void *recvbuf,
                                          const int *recvcounts, const int *displs,
                                          MPI_Datatype recvtype, MPIR_Comm * comm,
                                          MPIR_TSP_sched_t * sched)
{
    int mpi_errno = MPI_SUCCESS;
    int size, is_inplace, rank;
    int i, j, jnext, left, right;
    int recv_id = -1;
    int vtcs[2], r_vtcs[3];
    int nvtcs = 0;
    int tag;

    size_t recvtype_lb, recvtype_extent;
    size_t recvtype_true_extent;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIR_TSP_IALLGATHERV_SCHED_INTRA_RING);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIR_TSP_IALLGATHERV_SCHED_INTRA_RING);

    is_inplace = (sendbuf == MPI_IN_PLACE);
    size = MPIR_Comm_size(comm);
    rank = MPIR_Comm_rank(comm);

    /* find out the buffer which has the send data and point data_buf to it */
    if (is_inplace) {
        sendcount = recvcounts[rank];
        sendtype = recvtype;
    }

    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_lb, &recvtype_true_extent);
    recvtype_extent = MPL_MAX(recvtype_extent, recvtype_true_extent);

    if (!is_inplace) {
        /* copy your data into your recvbuf from your sendbuf */
        vtcs[0] = MPIR_TSP_sched_localcopy((char *) sendbuf, sendcount, sendtype,
                                           (char *) recvbuf + displs[rank] * recvtype_extent,
                                           recvcounts[rank], recvtype, sched, 0, NULL);
        nvtcs = 1;
    }

    left = (size + rank - 1) % size;
    right = (rank + 1) % size;

    j = rank;
    jnext = left;

    for (i = 0; i < size - 1; i++) {
        /* New tag for each send-recv pair. */
        mpi_errno = MPIR_Sched_next_tag(comm, &tag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        MPIR_TSP_sched_isend(((char *) recvbuf + displs[j] * recvtype_extent), recvcounts[j],
                             recvtype, right, tag, comm, sched, nvtcs, vtcs);

        recv_id =
            MPIR_TSP_sched_irecv((char *) recvbuf + displs[jnext] * recvtype_extent,
                                 recvcounts[jnext], recvtype, left, tag, comm, sched, i > 3 ? 1 : 0,
                                 r_vtcs + i % 3);
        r_vtcs[i % 3] = recv_id;
        vtcs[0] = recv_id;
        nvtcs = 1;

        j = jnext;
        jnext = (size + jnext - 1) % size;
    }

    MPIR_TSP_sched_fence(sched);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIR_TSP_IALLGATHERV_SCHED_INTRA_RING);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


/* Non-blocking ring based Allgatherv */
#undef FUNCNAME
#define FUNCNAME MPIR_TSP_Iallgatherv_intra_ring
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_TSP_Iallgatherv_intra_ring(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                    void *recvbuf, const int *recvcounts, const int *displs,
                                    MPI_Datatype recvtype, MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_TSP_sched_t *sched;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIR_TSP_IALLGATHERV_INTRA_RING);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIR_TSP_IALLGATHERV_INTRA_RING);

    *req = NULL;

    /* generate the schedule */
    sched = MPL_malloc(sizeof(MPIR_TSP_sched_t), MPL_MEM_COLL);
    MPIR_Assert(sched != NULL);
    MPIR_TSP_sched_create(sched);

    mpi_errno =
        MPIR_TSP_Iallgatherv_sched_intra_ring(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                                              displs, recvtype, comm, sched);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    /* start and register the schedule */
    mpi_errno = MPIR_TSP_sched_start(sched, comm, req);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIR_TSP_IALLGATHERV_INTRA_RING);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
