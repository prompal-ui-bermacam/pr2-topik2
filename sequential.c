#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main(int argc, char** argv)
{
    int N = atoi(argv[1]);

    long double a[N][N], b[N][N], c[N][N];
    int i,j,k;
    clock_t start, end;
    double cpu_time;
    for(i=0;i<N;i++){
	    for(j=0;j<N;j++){
	        b[i][j]=i;
	        a[i][j]=j;
	        c[i][j]=0;
	    }
    }
	
	start = clock();
    for(i=0;i<N;i++){
	    for(j=0;j<N;j++){
	    	for(k=0;k<N;k++){
	            c[i][j]+=a[i][k]*b[k][j];
	    	}
	    }
    }

    // for(i=0;i<N;i++){
    //     for(j=0;j<N;j++){
    //         printf("%d ",c[i][j]);
    //     }
    //     printf("\n");
    // }
    end = clock();

    cpu_time= ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("\nCPU time=%f sec",cpu_time);
	return 0;
}