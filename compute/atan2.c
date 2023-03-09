// SPDX-License-Identifier: MIT
/* Copyright 2022 Eileen Yoon <eyn@gmx.com> */

#include "ane.h"
#include "ane_arr.h"
#include "anec_atan2.h"

#include <time.h>

int main(void)
{
	int err = 0;
	struct ane_nn *nn = ane_init_atan2();
	if (nn == NULL){
		printf("error\n");
		return -1;
	}

	struct ane_arr *arr_a = ane_arr_init(nn, 0);
	struct ane_arr *arr_b = ane_arr_init(nn, 1);

	ane_fread(arr_a->data, arr_data_size(arr_a), "A.arr");
	ane_fread(arr_b->data, arr_data_size(arr_b), "B.arr");

	ane_arr_tile(arr_a);
	ane_arr_tile(arr_b);

	void *output = ane_zmemalign(output_size(nn, 0));


	clock_t tic = clock();

	ane_send(nn, arr_a->tile, 0);
	ane_send(nn, arr_b->tile, 1);

	err = ane_exec(nn);

	ane_read(nn, output, 0);

	clock_t toc = clock();
	printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);


	ane_fwrite(output, output_size(nn, 0), "atan2.tile");
	free(output);

	ane_arr_free(arr_a);
	ane_arr_free(arr_b);

	ane_free(nn);

	return err;
}
