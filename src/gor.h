#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <cassert>
#include <optional>
#include <variant>
// #include "types_gor.h"
#include "graph.h"
// #include "parser_dh.h"


typedef  /* arc */
   struct arc_st
{
   long              len;            /* length of the arc */
   struct node_st   *head;           /* head node */
}
  arc;

typedef  /* node */
   struct node_st
{
   arc              *first;           /* first outgoing arc */
   long              dist;	      /* tentative shortest path length */
   struct node_st   *parent;          /* parent pointer */
   arc              *current;         /* current arc in DFS */
   int               status;          /* status of node */
   int               temp;            /* for temporary labels */
} node;




int gor ( long n, node* nodes, node* source, const Distances& potential );

int parse(node** nodes, arc** arcs, node** nodes_ad, long* nmin, const Graph& graph);

std::optional<Distances> GOR(Graph& graph, NodeID source, const Distances& potential = Distances());

std::optional<Distances> gor(Graph &G, NodeID source);
std::optional<Distances> gor(Graph &G, const Distances& potential);