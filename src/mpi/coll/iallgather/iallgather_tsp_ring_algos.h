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

/* Header protection (i.e., IALLGATHER_TSP_RING_ALGOS_H_INCLUDED) is
 * intentionally omitted since this header might get included multiple
 * times within the same .c file. */

#include "tsp_namespace_def.h"

/* Algorithm: schedule a ring based allgather
 *
 * In the first step, each process i sends its contribution to process
 * i+1 and receives the contribution from process i-1 (with
 * wrap-around).  From the second step onwards, each process i
 * forwards to process i+1 the data it received from process i-1 in
 * the previous step.  This takes a total of p-1 steps.
 *
 * Cost = (p-1).alpha + n.((p-1)/p).beta
 *
 * This algorithm is preferred to recursive doubling for long messages
 * and small number of ranks.
 */
#undef FUNCNAME
#define FUNCNAME MPIR_TSP_Iallgather_sched_intra_ring
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_TSP_Iallgather_sched_intra_ring(const void *sendbuf, int sendcount,
                                         MPI_Datatype sendtype, void *recvbuf,
                                         int recvcount, MPI_Datatype recvtype,
                                         MPIR_Comm * comm, MPIR_TSP_sched_t * sched)
{
    int mpi_errno = MPI_SUCCESS;
    int i, j, jnext, left, right;
    int recv_id = -1;
    int vtcs[2], r_vtcs[3];
    int nvtcs = 0;

    int size = MPIR_Comm_size(comm);
    int rank = MPIR_Comm_rank(comm);
    int is_inplace = (sendbuf == MPI_IN_PLACE);
    int tag;

    size_t recvtype_lb, recvtype_extent;
    size_t sendtype_lb, sendtype_extent;
    size_t sendtype_true_extent, recvtype_true_extent;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIR_TSP_IALLGATHER_SCHED_INTRA_RING);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIR_TSP_IALLGATHER_SCHED_INTRA_RING);

    /* Find out the buffer which has the send data and point data_buf to it */
    if (is_inplace) {
        sendcount = recvcount;
        sendtype = recvtype;
    }

    /* Get datatype info of sendtype and recvtype */
    MPIR_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPIR_Type_get_true_extent_impl(sendtype, &sendtype_lb, &sendtype_true_extent);
    sendtype_extent = MPL_MAX(sendtype_extent, sendtype_true_extent);

    MPIR_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_lb, &recvtype_true_extent);
    recvtype_extent = MPL_MAX(recvtype_extent, recvtype_true_extent);

    if (!is_inplace) {
        /* Copy your data into your recvbuf from your sendbuf */
        vtcs[0] = MPIR_TSP_sched_localcopy((char *) sendbuf, sendcount, sendtype,
                                           (char *) recvbuf + rank * recvcount * recvtype_extent,
                                           recvcount, recvtype, sched, 0, NULL);

        nvtcs = 1;
    }

    /* In ring algorithm src and dst are fixed */
    left = (size + rank - 1) % size;
    right = (rank + 1) % size;

    j = rank;
    jnext = left;

    /* Ranks pass around the data (size - 1) times */
    /* irecv in reality has no dependence to anything, therefore potentially
     * they can be issued all together when executing the scheduler, which
     * may not be a good thing. Use r_vtcs to limit the number of posted recv
     * to 3
     */
    for (i = 1; i < size; i++) {

        mpi_errno = MPIR_Sched_next_tag(comm, &tag);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        MPIR_TSP_sched_isend(((char *) recvbuf + j * recvcount * recvtype_extent),
                             recvcount, recvtype, right, tag, comm, sched, nvtcs, vtcs);

        recv_id = MPIR_TSP_sched_irecv(((char *) recvbuf + jnext * recvcount * recvtype_extent),
                                       recvcount, recvtype, left, tag, comm, sched, i > 3 ? 1 : 0,
                                       r_vtcs + i % 3);
        r_vtcs[i % 3] = recv_id;
        MPL_DBG_MSG_FMT(MPIR_DBG_COLL, VERBOSE,
                        (MPL_DBG_FDEST, "posting recv at address=%p, count=%d",
                         ((char *) recvbuf + jnext * recvcount * recvtype_extent),
                         size * recvcount));

        vtcs[0] = recv_id;
        nvtcs = 1;

        j = jnext;
        jnext = (size + jnext - 1) % size;
    }

  fn_exit:
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIR_TSP_IALLGATHERV_SCHED_INTRA_RING);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

/* Non-blocking ring based Allgather */
#undef FUNCNAME
#define FUNCNAME MPIR_TSP_Iallgather_intra_ring
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_TSP_Iallgather_intra_ring(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                                   MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_TSP_sched_t *sched;
    *req = NULL;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIR_TSP_IALLGATHER_INTRA_RING);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIR_TSP_IALLGATHER_INTRA_RING);


    /* Generate the schedule */
    sched = MPL_malloc(sizeof(MPIR_TSP_sched_t), MPL_MEM_COLL);
    MPIR_ERR_CHKANDJUMP(!sched, mpi_errno, MPI_ERR_OTHER, "**nomem");
    MPIR_TSP_sched_create(sched);

    mpi_errno =
        MPIR_TSP_Iallgather_sched_intra_ring(sendbuf, sendcount, sendtype, recvbuf,
                                             recvcount, recvtype, comm, sched);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    /* Start and register the schedule */
    mpi_errno = MPIR_TSP_sched_start(sched, comm, req);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIR_TSP_IALLGATHER_INTRA_RING);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
