// SPDX-License-Identifier: MIT
/* Copyright 2022 Eileen Yoon <eyn@gmx.com> */

#include "ane.h"
#include "ane_buf.h"
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

	struct ane_buf *buf1 = ane_buf_init_input(nn, 0);
	struct ane_buf *buf2 = ane_buf_init_input(nn, 1);
	struct ane_buf *buf3 = ane_buf_init_output(nn, 0);

	ane_fread(buf1->data, buf1->data_size, "A.arr");
	ane_fread(buf2->data, buf2->data_size, "B.arr");

	ane_buf_tile(buf1);
	ane_buf_tile(buf2);

#if 1
	clock_t tic = clock();

	ane_buf_send(nn, buf1);
	ane_buf_send(nn, buf2);

	err = ane_exec(nn);

	ane_buf_read(nn, buf3);

	clock_t toc = clock();

	printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

	ane_buf_untile(buf3);
	ane_fwrite(buf3->data, buf3->data_size, "atan2.arr");
#endif

	ane_buf_free(buf3);
	ane_buf_free(buf2);
	ane_buf_free(buf1);

	ane_free(nn);

	return err;
}
