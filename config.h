#ifndef __CONFIG_H
#define __CONFIG_H

#define NB_CHARS 67							// bin count 
#define NB_BLOCKS 32						// number of blocks to be used for calculation
#define NB_THREADS 1024						// number of threads per block used for calculation
#define TOTAL_THREAD NB_BLOCKS*NB_THREADS	// total thread count

#define ASCII_DOLLAR 36						// to not use the ascii of $
#define ASCII_ASTERIK 42					// to not use the ascii of *

#define CHAR_DIFF_BET_36_42 1   			// to fix the histogram position between chars $ and *
#define CHAR_DIFF_ABOVE_42 2    			// to fix the histogram position after char *

#define MIN_ASCII 32						// min ascii used for the program
#define MAX_ASCII 126						// max ascii used for the program

#define UPPERCASE_ALPHABET 65				// min ascii of uppercase letter
#define SKIP_UPPERCASE 58					// count of uppercase letters + MIN_ASCII 

#endif