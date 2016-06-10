/*
 * This defines and trains an ensemble of neural networks.
 */

#ifndef ENSEMBLE_H
#define ENSEMBLE_H

#include <QList>
#include "Network.h"
#include "ProblemInfo.h"

class NetworkEnsemble
{
public:
    explicit NetworkEnsemble(int);
    virtual ~NetworkEnsemble();
    
    /*
     * For training we need two lists: the first contains training samples, the second a subset of test samples
     * used to measure a network's performance between two epochs.
     */
    void training(QList< InputSample* >&, QList< InputSample* > &);
    
    /*
     * Tests the performance of a network, printing out some results on the command line. Returns the percentage
     * of right answers given on the set.
     */
    double test(QList< InputSample* >&);
    
private:
    QList< Network* > m_networks;
    int m_nextId; /* next available ID for a network */
    
    /*
     * Finds how is the network performing, as a percentage of wrong answers over all the test set.
     */
    double computeAverageError(Network *, const QList< InputSample* > &);
    
    /*
     * Functions needed for NSGA-II.
     */
    QMap< int, QList< Network* > > computeParetoFrontRank(QList< Network* >);
    QList< Network* > paretoFront(QList< Network* > &);
    bool paretoDominates(Network *, Network *);
    QList< Network* > breed(QList< Network* >);
    QList< Network* > sortBySparsity(QList< Network* >);
    
    /*
     * Functions needed to sort the network list when computing the sparsity of each network.
     */
    static bool lessThanError(const Network *, const Network *);
    static bool lessThanComplexity(const Network *, const Network *);
    static bool lessThanSparsity(const Network *, const Network *);
};

#endif