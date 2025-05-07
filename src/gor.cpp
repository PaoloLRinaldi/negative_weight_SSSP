// #pragma once

// #include <iostream>
// #include <vector>
// #include <cstdlib>
// #include <string>
// #include <cassert>
// #include "types_gor.h"
// #include "graph.h"
// #include "parser_dh.h"
#include "gor.h"


typedef enum { OUT_OF_STACKS = 0, IN_NEW_PASS, IN_TOP_SORT, IN_PASS } status;

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
       *arc_first = nullptr,              /* internal array for holding
                                     - node degree
                                     - position of the first outgoing arc */
       *arc_tail = nullptr,               /* internal array: tails of the arcs */
        /* temporary variables carrying no of nodes */
        head, tail, i;

long    m,                      /* internal number of arcs */
        /* temporary variable carrying no of arcs */
        last, arc_num, arc_new_num;

node    *head_p = nullptr;

arc     *arc_current = nullptr,
        *arc_new = nullptr;

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

                if ( *nodes == nullptr || *arcs == nullptr || 
                     arc_first == nullptr || arc_tail == nullptr )
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
if (arc_first != nullptr) {
  std::free ( arc_first );
  arc_first = nullptr;
}
if (arc_tail != nullptr) {
  std::free ( arc_tail );
  arc_tail = nullptr;
}

/* Uff! all is done */
return (0);

/* ---------------------------------- */
 error:  /* error found reading input */

  std::cerr << "Prs" << err_no << ": line " << no_lines << " of input - " << err_message[err_no] << std::endl;
exit (1);

}
/* --------------------   end of parser  -------------------*/


int gor ( long n, node* nodes, node* source, const Distances& potential )

{

#define NNULL nullptr
#define VERY_FAR  1073741823

/* -----   stacks definitions ----- */

typedef struct str_stack
           {
	     long     top;
             node   **arr = nullptr;
	   }
             stack;

stack      new_pass, pass, top_sort;

/* status of node regarding to stacks */ 

// #define OUT_OF_STACKS  0
// #define IN_NEW_PASS    1
// #define IN_TOP_SORT    2
// #define IN_PASS        3

#define INIT_STACKS( source )\
{\
  new_pass.arr = (node**) std::calloc ( n + 1, sizeof (node*) );\
  new_pass.arr[0] = source;\
  new_pass.top    = 0;\
  source -> status = IN_NEW_PASS;\
\
  pass.arr = (node**) std::calloc ( n + 1, sizeof (node*) );\
  pass.top = -1;\
\
  top_sort.arr = (node**) std::calloc ( n + 1, sizeof (node*) );\
  top_sort.top = -1;\
\
  if ( new_pass.arr == nullptr || \
       pass.arr     == nullptr || \
       top_sort.arr == nullptr    \
     )\
    return ( 1 );\
}

#define FREE_STACKS()\
{\
if ( new_pass.arr != nullptr ) {\
  std::free ( new_pass.arr );\
  new_pass.arr = nullptr;\
}\
if ( pass.arr != nullptr ) {\
  std::free ( pass.arr );\
  pass.arr = nullptr;\
}\
if ( top_sort.arr != nullptr ) {\
  std::free ( top_sort.arr );\
  top_sort.arr = nullptr;\
}\
}

#define NONEMPTY_STACK( stack )     ( stack.top >= 0 )

#define POP( stack, node )\
{\
node = stack.arr [ stack.top ];\
stack.top -- ;\
}

#define PUSH( stack, node )\
{\
  stack.top ++ ;\
  stack.arr [ stack.top ] = node;\
}\

/* -------------------------------------- */

long dist_new,
     dist_i;

node *node_last,
     *i,
     *j;

arc  *arc_ij,
     *arc_last;

long num_scans = 0;

/* initialization */

node_last = nodes + n ;

// bool using_potential = potential.size() == n;

// if (!using_potential) {
  
  for ( i = nodes; i != node_last; i ++ )
    { 
        i -> parent   = NNULL;
        i -> dist     = VERY_FAR;
        i -> status   = OUT_OF_STACKS;
    }

  source -> parent = source;
  source -> dist   = 0;

// } else {
//   NodeID k = 0;
 
//   for ( i = nodes; i != node_last; i ++, k++ )
//     { 
//         i -> parent   = NNULL;
//         i -> dist     = potential[k];
//         // i -> status   = IN_NEW_PASS;
//         i -> status   = OUT_OF_STACKS;
//     }

//   source = nodes;  // I set the source to be the first node

// }

INIT_STACKS (source)


/* main loop */

while ( NONEMPTY_STACK ( new_pass ) )
{
/* topological sorting */

  while ( NONEMPTY_STACK ( new_pass ) )
    {
      POP ( new_pass, i )

      if (  i -> status == IN_NEW_PASS )
	{ /* looking for any arc with negative reduced cost outgoing from i
             if any - start deapth first search */

          arc_last = ( i + 1 ) -> first;
          dist_i   = i -> dist;
   
          for ( arc_ij = i -> first; arc_ij != arc_last; arc_ij ++ ) {
            // try {
            //   if ( dist_i + arc_ij -> len
            //       <
            //       ( arc_ij -> head ) -> dist
            //     )
            //  	 break;
            // } catch (std::exception& e) {
            //   std::cout << "arcij " << arc_ij << " head " << arc_ij -> head << std::endl << std::flush;
            // }
            //   if (arc_ij == (arc*)0x000000000008) std::cout << "Arc not valid:" << std::endl << std::flush;
            //   if ((arc_ij -> head) == (node*)0x000000000008) std::cout << "First not valid" << std::endl << std::flush;
              if (!(arc_ij -> head)) {
                // std::cout << "First not valid" << std::endl << std::flush;
              // std::cout << "Head " << arc_ij -> head << std::endl << std::flush;
              }
             if ( dist_i + arc_ij -> len
                  <
                  ( arc_ij -> head ) -> dist
                )
             	 break;
          }
	
          if ( arc_ij != arc_last )
            {
              i -> status  = IN_TOP_SORT;
              i -> current = i -> first;
	    }
          else
              i -> status = OUT_OF_STACKS;
	}


      if ( i -> status == IN_TOP_SORT )
	{ /* deapth first search */

         while ( 1 )
           {        
            arc_last = ( i + 1 ) -> first;
            dist_i   = i -> dist;

            for ( arc_ij = i -> current; arc_ij != arc_last; arc_ij ++ )
	      { 
	        j = arc_ij -> head;

                if ( dist_i + arc_ij -> len  <=  j -> dist &&
                     /* j -> dist   < VERY_FAR                && */
                     j -> status < IN_TOP_SORT
                   )
		    {
		      i -> current = arc_ij + 1;
                      PUSH ( top_sort, i )
		      j -> status = IN_TOP_SORT;
                      j -> current = j -> first;
		      i = j;

		      break;
		    }
	      }
	    
	    if ( arc_ij == arc_last )
              {
		i -> status = IN_PASS;
		PUSH ( pass, i );
                num_scans ++;

	        if ( NONEMPTY_STACK ( top_sort ) )
		    POP ( top_sort, i )
		else
		    break;
	      }
	  } /* end of deapth first search */
       }


    } /* end of topological sorting */
	      
/* Bellman - Ford pass */

  while ( NONEMPTY_STACK ( pass ) )
   {
      num_scans ++;

      POP ( pass, i )
      i -> status = OUT_OF_STACKS;

      arc_last = ( i + 1 ) -> first;
      dist_i   = i -> dist;

      for ( arc_ij = i -> first; arc_ij != arc_last; arc_ij ++ )
         { /* scanning arcs outgoing from  i  */
           j  = arc_ij -> head;

           dist_new = dist_i + ( arc_ij -> len );

           if ( dist_new <  j -> dist )
	     { 
	         j -> dist   = dist_new;
                 j -> parent = i;

                 if (  j -> status == OUT_OF_STACKS  )
		   {
		     PUSH ( new_pass, j )
		     j -> status = IN_NEW_PASS;

		   }
	     }
	 
           } /* end of scanning  i  */
    } /* end of one pass */

 } /* end of the main loop */

    // std::cout << "Valid size 2" << std::endl << std::flush;
  FREE_STACKS ();
  // std::cout << "Invalid size 2" << std::endl << std::flush;

return ( 0 );
}


std::optional<Distances> GOR(Graph& graph, NodeID source, const Distances& potential) {

    NodeID n = graph.numberOfNodes();
    const EdgeID m = graph.numberOfEdges();
    node *ndp = nullptr, *nodes = nullptr, *source_node = nullptr;
    long nmin;
    arc *arcs = nullptr;

    // std::cout << "I get here" << std::endl << std::flush;

    // if (potential.size() == n) {
    //   source = graph.addSuperSource_not_reversed(potential);
    //   n = source;
    // }

    // std::cout << "I get here 2" << std::endl << std::flush;

    parse(&nodes, &arcs, &ndp, &nmin, graph);
    source_node = nodes + source;

    // std::cout << "I get here 3" << std::endl << std::flush;

//     std::cout << "n= " << n << ", m= " << m << ", nmin= " << nmin << ", source = " << (source_node-ndp)+nmin << std::endl << std::flush;

//   std::cout << "ordered arcs:" << std::endl << std::flush;
//   {
//     int i;
//     for (node *k = ndp; k < ndp + n; k++) {
//         i = (k-ndp)+nmin;
//         for (arc *ta = k -> first; ta != (k+1)-> first; ta++) {
//             std::cout << i << " " << ((ta->head)-ndp)+nmin << " " << ta->len << std::endl << std::flush;
//         }
//     }
//   }


    gor ( n, ndp, source_node, Distances() );

    // std::cout << "I get here 4" << std::endl << std::flush;

    Distances distances(n, c::infty);

    NodeID i = 0;

    for (node* k= ndp; k< ndp + n; k++, i++ )
        if ( k -> parent != nullptr )
            distances[i] = k -> dist;

    // std::cout << "Valid size" << std::endl << std::flush;
    if (nodes != nullptr) {
        std::free(nodes);
        nodes = nullptr;
    }
    if (arcs != nullptr) {
        std::free(arcs);
        arcs = nullptr;
    }
    // std::cout << "Invalid size" << std::endl << std::flush;

    // if (potential.size() == n - 1) {
    //   graph.removeSuperSource_not_reversed();
    //   distances.erase(distances.end() - 1, distances.end());
    // }

    // std::cout << "I get here 5" << std::endl << std::flush;

    return distances;

}

std::optional<Distances> gor(Graph &G, std::variant<NodeID, const Distances*> source_pot) {
  NodeID n = G.numberOfNodes();
  NodeID num_scans = 0;
  std::vector<NodeID> cnt(n, 0);  // counts the number of updates for each node

  Distances dist(n, c::infty);
  const Distances *potential;
  bool using_potential = std::holds_alternative<const Distances*>(source_pot);
  std::vector<EdgeRange::iterator> Current(n);
  std::vector<EdgeRange> edgesOf(n);
  for (NodeID i = 0; i < n; i++)
    edgesOf[i] = G.getEdgesOf(i, Orientation::OUT);
  std::vector<status> Status(n, OUT_OF_STACKS);


  // initialize stacks

  std::vector<NodeID> pass;
  std::vector<NodeID> new_pass;
  std::vector<NodeID> top_sort;
  try {
    pass.reserve(n);
    new_pass.reserve(n);
    top_sort.reserve(n);
  } catch (...) {
    throw 1;
  }
  if (using_potential) {
    potential = std::get<const Distances*>(source_pot);
    if (potential->size() != n) {
      throw 1;
    }
    for (NodeID i = 0; i < n; i++) {
      dist[i] = -(*potential)[i];
      new_pass.push_back(i);
      Status[i] = IN_NEW_PASS;
    }
  } else {
    NodeID source = std::get<NodeID>(source_pot);
    dist[source] = 0;
    new_pass.push_back(source);
    Status[source] = IN_NEW_PASS;
  }

  while (new_pass.size() != 0) {  // main loop
    NodeID i;
    NodeID j;

    while (new_pass.size() != 0) {  // topological sorting
      i = new_pass.back();
      new_pass.pop_back();

      assert(top_sort.size() == 0);
      if (Status[i] == IN_NEW_PASS) {
        /* We look for any arc with negative reduced cost in \f$FS(i)\f$.
           If found, we start deapth first search */
        bool found = false;
        // for (const Edge &arc_ij : G.getEdgesOf(i, Orientation::OUT))
        for (const Edge &arc_ij : edgesOf[i]) {
          auto pot_edge_w = arc_ij.weight;
          if (using_potential) {
            pot_edge_w += (*potential)[i] - (*potential)[arc_ij.target];
          }
          if (dist[i] + pot_edge_w < dist[arc_ij.target] &&
            dist[i] < c::infty) {
            found = true;
            break;
          }
        }

        if (found) {
          Status[i] = IN_TOP_SORT;
          // Current[i] = G.getEdgesOf(i, Orientation::OUT).begin();
          Current[i] = edgesOf[i].begin();
          top_sort.push_back(i);
        } else
          Status[i] = OUT_OF_STACKS;
      }

      while (top_sort.size() != 0) {  // deapth first search
        i = top_sort.back();
        top_sort.pop_back();
        bool found = false;
        for (auto arc_ij = Current[i];
            //  arc_ij != G.getEdgesOf(i, Orientation::OUT).end() && !found;
             arc_ij != edgesOf[i].end() && !found;
             ++arc_ij) {
          j = arc_ij->target;

          auto pot_edge_w = arc_ij->weight;
          if (using_potential) {
            pot_edge_w += (*potential)[i] - (*potential)[j];
          }

          if ((dist[i] < c::infty || pot_edge_w == 0) &&
               dist[i] + pot_edge_w <= dist[j] &&
               Status[j] < IN_TOP_SORT) {
            Current[i] = arc_ij;
            ++Current[i];
            top_sort.push_back(i);
            Status[j] = IN_TOP_SORT;
            // Current[j] = G.getEdgesOf(j, Orientation::OUT).begin();
            Current[j] = edgesOf[j].begin();
            i = j;
            found = true;
          }
        }
        if (found)
          top_sort.push_back(i);
        else {
          Status[i] = IN_PASS;
          pass.push_back(i);
          num_scans++;
	  //	  std::cout << num_scans << " " << i << std::endl;
        }

      }  // end of deapth first search
    }    // end of topological sorting

    // Bellman-Ford pass
    while (pass.size() != 0) {
      num_scans++;

      NodeID i = pass.back();
      pass.pop_back();
      Status[i] = OUT_OF_STACKS;
      //      std::cout << num_scans << " " << i << std::endl;

      // scanning \f$FS(i)\f$
      assert(dist[i] < c::infty);
      // for (auto arc_ij : G.getEdgesOf(i, Orientation::OUT)) {
      for (auto arc_ij : edgesOf[i]) {
        j = arc_ij.target;

        auto pot_edge_w = arc_ij.weight;
        if (using_potential) {
          pot_edge_w += (*potential)[i] - (*potential)[j];
        }

        Distance dist_new = dist[i] + pot_edge_w;

        if (dist_new < dist[j]) {
          dist[j] = dist_new;

          if (Status[j] == OUT_OF_STACKS) {
            new_pass.push_back(j);
            Status[j] = IN_NEW_PASS;
            cnt[j]++;
            if (cnt[j] > n) {  // brutal way to detect negative cycle
                return {};
            }
          }
        }
      }  // end of scanning \f$FS(i)\f$

    }  // end of one pass

  }  // end of main loop

  // std::cout << "n-scans: " << num_scans << std::endl;
  return dist;
}

std::optional<Distances> gor(Graph &G, NodeID source) {
  return gor(G, std::variant<NodeID, const Distances*>(source));
}

std::optional<Distances> gor(Graph &G, const Distances& potential) {
  return gor(G, std::variant<NodeID, const Distances*>(&potential));
}