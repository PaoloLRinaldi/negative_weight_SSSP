/* defs.h */
#pragma once

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
