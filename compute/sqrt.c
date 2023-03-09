// SPDX-License-Identifier: MIT
/* Copyright 2022 Eileen Yoon <eyn@gmx.com> */

#include "ane.h"
#include "ane_arr.h"
#include "anec_sqrt.h"

#include <time.h>

int main(void)
{
	int err = 0;
	struct ane_nn *nn = ane_init_sqrt();
	if (nn == NULL){
		return -1;
	}

	struct ane_arr *arr = ane_arr_init(nn, 0);
	ane_fread(arr->data, arr_data_size(arr), "A.arr");
	ane_arr_tile(arr);

	void *output = ane_zmemalign(output_size(nn, 0));


	clock_t tic = clock();

	ane_send(nn, arr->tile, 0);

	err = ane_exec(nn);

	ane_read(nn, output, 0);

	clock_t toc = clock();
	printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);


	ane_arr_free(arr);

	ane_fwrite(output, output_size(nn, 0), "sqrt.tile");
	free(output);

	ane_free(nn);

	return err;
}
