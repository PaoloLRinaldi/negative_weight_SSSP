/********************************************************************/
/*                                                                  */
/*  parse (...) :                                                   */
/*       1. Reads shortest path problem in extended DIMACS format.  */
/*       2. Prepares internal data representation #1.               */
/*                                                                  */
/********************************************************************/
       
/* files to be included: */

/*
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "types_dh.h"
*/

/*
   types: 'arc' and 'node' must be predefined

   type    arc    must contain fields 'len' and 'head': 

   typedef 
     struct arc_st
       {
         long            len;      .. length of the arc 
         struct node_st *head;     .. head node of the arc
         ....................
       } 
   arc;

   type    arc    must contain the field 'first': 

   typedef
     struct node_st
       {
          arc_st        *first;    ..  first outgoing arc 
          ....................
       }
    node;
*/

/* ----------------------------------------------------------------- */

#pragma once

#include "graph.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <cassert>


int parse(node** nodes, arc** arcs, node** nodes_ad, long* nmin, const Graph& graph)

{

#define MAXLINE       100	/* max line length in the input file */
#define ARC_FIELDS      3	/* no of fields in arc input  */
#define P_FIELDS        3       /* no of fields in problem line */
#define PROBLEM_TYPE "sp"       /* name of problem type*/
#define DEFAULT_NAME "unknown"  /* default name of the problem */

using FullEdges = std::vector<FullEdge>;

long    node_min,               /* minimal no of node  */
        node_max = 0,               /* maximal no of nodes */
       *arc_first,              /* internal array for holding
                                     - node degree
                                     - position of the first outgoing arc */
       *arc_tail,               /* internal array: tails of the arcs */
        /* temporary variables carrying no of nodes */
        head, tail, i;

long    m,                      /* internal number of arcs */
        /* temporary variable carrying no of arcs */
        last, arc_num, arc_new_num;

node    *head_p;

arc     *arc_current,
        *arc_new;

long    length;                 /* length of the current arc */

long    no_lines=0,             /* no of current input line */
        no_plines=0,            /* no of problem-lines */
        no_tlines=0,            /* no of title(problem name)-lines */
        no_nlines=0,            /* no of node(source)-lines */
        no_alines=0;            /* no of arc-lines */

char    in_line[MAXLINE],       /* for reading input line */
        pr_type[3];             /* for reading type of the problem */

int     k,                      /* temporary */
        err_no;                 /* no of detected error */

/* -------------- error numbers & error messages ---------------- */
#define EN1   0
#define EN2   1
#define EN3   2
#define EN4   3
#define EN6   4
#define EN10  5
#define EN7   6
#define EN8   7
#define EN9   8
#define EN11  9
#define EN12 10
#define EN13 11
#define EN14 12
#define EN16 13
#define EN15 14
#define EN17 15
#define EN18 16
#define EN21 17
#define EN19 18
#define EN20 19
#define EN22 20

static std::string err_message[] = 
  { 
/* 0*/    "more than one problem line.",
/* 1*/    "wrong number of parameters in the problem line.",
/* 2*/    "it is not a Shortest Path problem line.",
/* 3*/    "bad value of a parameter in the problem line.",
/* 4*/    "can't obtain enough memory to solve this problem.",
/* 5*/    "more than one line with the problem name.",
/* 6*/    "can't read problem name.",
/* 7*/    "problem description must be before source description.",
/* 8*/    "this parser doesn't support multiply sources.",
/* 9*/    "wrong number of parameters in the source line.",
/*10*/    "wrong value of parameters in the source line.",
/*11*/    "this parser doesn't support destination description.",
/*12*/    "source description must be before arc descriptions.",
/*13*/    "too many arcs in the input.",
/*14*/    "wrong number of parameters in the arc line.",
/*15*/    "wrong value of parameters in the arc line.",
/*16*/    "unknown line type in the input.",
/*17*/    "reading error.",
/*18*/    "not enough arcs in the input.",
/*19*/    "source doesn't have output arcs.",
/*20*/    "can't read anything from the input file."
  };
/* --------------------------------------------------------------- */

/* The main loop:
        -  reads the line of the input,
        -  analises its type,
        -  checks correctness of parameters,
        -  puts data to the arrays,
        -  does service functions
*/
FullEdges edges;
NodeID n = graph.numberOfNodes();
for (NodeID v = 0; v < n; v++) {
    for (auto const& e : graph.getEdgesOf(v)) {
        edges.emplace_back(v, e.target, e.weight);
    }
}

node_min = (long) n;

int a;

m = (long) edges.size();
        /* allocating memory for  'nodes', 'arcs'  and internal arrays */
                *nodes    = (node*) std::calloc ( n+2, sizeof(node) );
		*arcs     = (arc*)  std::calloc ( m+1, sizeof(arc) );
	        arc_tail = (long*) std::calloc ( m,   sizeof(long) ); 
		arc_first= (long*) std::calloc ( n+2, sizeof(long) );
                /* arc_first [ 0 .. n+1 ] = 0 - initialized by calloc */

                if ( *nodes == NULL || *arcs == NULL || 
                     arc_first == NULL || arc_tail == NULL )
                    /* memory is not allocated */
		    { err_no = EN6; goto error; }

		/* setting pointer to the current arc */
		arc_current = *arcs;

for (const auto& e : edges) {

    tail = std::get<0>(e);
    head = std::get<1>(e);
    length = std::get<2>(e);

    assert(tail < n);
    assert(head < n);

		if ( tail < 0  ||  tail > n  ||
                     head < 0  ||  head > n  
		   )
                    /* wrong value of nodes */
		    { err_no = EN17; goto error; }

		arc_first[tail + 1] ++; /* no of arcs outgoing from tail
                                           is stored in arc_first[tail+1] */

                /* storing information about the arc */
		arc_tail[no_alines] = tail;
		arc_current -> head = *nodes + head;
		arc_current -> len  = length;

		/* searching minimumu and maximum node */
                if ( head < node_min ) node_min = head;
                if ( tail < node_min ) node_min = tail;
                if ( head > node_max ) node_max = head;
                if ( tail > node_max ) node_max = tail;

		no_alines ++;
		arc_current ++;\
}     /* end of input loop */

/* ----- all is red  or  error while reading ----- */ 

if ( no_alines < m ) /* not enough arcs */
  { err_no = EN19; goto error; } 

  
/********** ordering arcs - linear time algorithm ***********/

/* first arc from the first node */
( *nodes + node_min ) -> first = *arcs;

/* before below loop arc_first[i+1] is the number of arcs outgoing from i;
   after this loop arc_first[i] is the position of the first 
   outgoing from node i arcs after they would be ordered;
   this value is transformed to pointer and written to node.first[i]
   */
 
for ( i = node_min + 1; i <= node_max + 1; i ++ ) 
  {
    if (i > n + 1 || i < 0) {
      // std::cout << "Parser: i over: " << i  << " " << n + 1 << std::endl << std::flush;
    }
    arc_first[i]          += arc_first[i-1];
    ( *nodes + i ) -> first = *arcs + arc_first[i];
  }


for ( i = node_min; i < node_max; i ++ ) /* scanning all the nodes  
                                            exept the last*/
  {

    last = ( ( *nodes + i + 1 ) -> first ) - *arcs;
                             /* arcs outgoing from i must be cited    
                              from position arc_first[i] to the position
                              equal to initial value of arc_first[i+1]-1  */

    for ( arc_num = arc_first[i]; arc_num < last; arc_num ++ )
      { tail = arc_tail[arc_num];

	while ( tail != i )
          /* the arc no  arc_num  is not in place because arc cited here
             must go out from i;
             we'll put it to its place and continue this process
             until an arc in this position would go out from i */

	  { arc_new_num  = arc_first[tail];
	    arc_current  = *arcs + arc_num;
	    arc_new      = *arcs + arc_new_num;
	    
	    /* arc_current must be cited in the position arc_new    
	       swapping these arcs:                                 */

	    head_p               = arc_new -> head;
	    arc_new -> head      = arc_current -> head;
	    arc_current -> head = head_p;

	    length              = arc_new -> len;
	    arc_new -> len      = arc_current -> len;
	    arc_current -> len = length;

	    arc_tail[arc_num] = arc_tail[arc_new_num];

	    /* we increase arc_first[tail] but label previous position */

	    arc_tail[arc_new_num] = tail;
	    arc_first[tail] ++ ;

            tail = arc_tail[arc_num];
	  }
      }
    /* all arcs outgoing from  i  are in place */
  }       

/* -----------------------  arcs are ordered  ------------------------- */

/* assigning output values */
*nodes_ad = *nodes + node_min;
*nmin = node_min;

/* free internal memory */
std::free ( arc_first ); std::free ( arc_tail );

/* Uff! all is done */
return (0);

/* ---------------------------------- */
 error:  /* error found reading input */

  std::cerr << "Prs" << err_no << ": line " << no_lines << " of input - " << err_message[err_no] << std::endl;
exit (1);

}
/* --------------------   end of parser  -------------------*/
