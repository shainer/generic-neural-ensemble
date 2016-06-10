/*
 * This class holds all the links in a network, indexed by the identifiers of the two neurons
 * connected by the link.
 * 
 * I didn't use a matrix because I need to add and remove neurons during training, and reallocating memory for a matrix
 * each time was too computationally-expensive.
 */

#ifndef LINKMATRIX_H
#define LINKMATRIX_H

#include <QMap>
#include <QPair>
#include <QHash>

#include "Link.h"

class LinkMatrix
{
public:
    explicit LinkMatrix();
    virtual ~LinkMatrix();
    
    /*
     * Returns the link between the neurons identified by the two parameters, or NULL
     * if none exists.
     */
    Link* link(int, int) const;
    
    /*
     * Returns all the pairs used as keys internally.
     */
    QList< QPair<int, int> > keys() const;
    
    /*
     * Returns a list with all the links.
     */
    QList< Link* > links() const;
    
    /*
     * Returns the number of links.
     */
    int complexity() const;
    
    /*
     * Add a new link between two neurons.
     */
    void addLink(int, int, Link *);
    
    /*
     * Remove a link between two neurons.
     */
    void removeLink(int, int);
    
    /*
     * Remove all links going to or coming from @neuron. @maxId is the maximum neuron ID currently active,
     * while the @neurons hash is needed to remove the link from inConnections or outConnections lists.
     */
    void removeAllLinks(int neuron, int maxId, const QHash< int, Neuron* >& neurons);
    
    /*
     * This must be called when changing the ID of a neuron: updates all the links concerning it.
     */
    void changeId(int oldId, int newId);
    
private:
    QMap< QPair<int, int>, Link* > m_links;
};

#endif