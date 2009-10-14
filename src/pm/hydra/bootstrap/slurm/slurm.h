/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2008 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef SLURM_H_INCLUDED
#define SLURM_H_INCLUDED

#include "hydra_base.h"

HYD_Status HYD_BSCD_slurm_launch_procs(char **global_args, const char *proxy_id_str,
                                       struct HYD_Proxy *proxy_list);
HYD_Status HYD_BSCD_slurm_query_proxy_id(int *proxy_id);
HYD_Status HYD_BSCD_slurm_query_node_list(int *num_nodes, struct HYD_Proxy **proxy_list);

#endif /* SLURM_H_INCLUDED */
