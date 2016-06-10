/*
 * This class holds all the links in a network, indexed by the identifiers of the two neurons
 * connected by the link.
 * 
 * I didn't use a matrix because I need to add and remove neurons during training, and reallocating memory for a matrix
 * each time was too computationally-expensive.
 */

#include "LinkMatrix.h"
#include "Neuron.h"

#include <QtCore/QDebug>

LinkMatrix::LinkMatrix()
{}

LinkMatrix::~LinkMatrix()
{
    qDeleteAll(m_links);
}

Link* LinkMatrix::link(int n1, int n2) const
{
    return m_links.value( qMakePair<int, int>(n1, n2) );
}

QList< QPair< int, int > > LinkMatrix::keys() const
{
    return m_links.keys();
}

QList< Link* > LinkMatrix::links() const
{
    return m_links.values();
}

int LinkMatrix::complexity() const
{
    return m_links.size();
}

void LinkMatrix::removeLink(int n1, int n2)
{
    m_links.remove( qMakePair<int, int>(n1, n2) );
}

void LinkMatrix::addLink(int n1, int n2, Link* link)
{
    m_links.insert( qMakePair<int, int>(n1, n2), link );
}

void LinkMatrix::removeAllLinks(int neuron, int maxId, const QHash< int, Neuron* >& neurons)
{
    for (int i = -1; i <= maxId; i++) {
        if (i == neuron) {
            continue;
        }
        
        Link* linkOut = link(neuron, i);
        Link* linkIn = link(i, neuron);
        
        if (linkOut) {
            neurons[i]->removeInConnection(linkOut);
            m_links.remove( qMakePair<int, int>(neuron, i) );
            
            delete linkOut;
            linkOut = 0;
        }
        
        if (linkIn) {
            neurons[i]->removeOutConnection(linkIn);
            m_links.remove( qMakePair<int, int>(i, neuron) );
            
            delete linkIn;
            linkIn = 0;
        }
    }
}

void LinkMatrix::changeId(int oldId, int newId)
{
    for (int i = -1; i < oldId; i++) {
        Link* linkIn = link(i, oldId);
        Link* linkOut = link(oldId, i);
        
        if (linkOut) {
            m_links.remove( qMakePair<int, int>(oldId, i) );
            m_links.insert( qMakePair<int, int>(newId, i), linkOut );
        }
        
        if (linkIn) {
            m_links.remove( qMakePair<int, int>(i, oldId) );
            m_links.insert( qMakePair<int, int>(i, newId), linkIn );
        }
    }
}
