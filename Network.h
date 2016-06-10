/*
 * Neural network for a binary classification problem.
 */

#ifndef NETWORK_H
#define NETWORK_H

#include "Neuron.h"
#include "Link.h"
#include "Utils.h"
#include "LinkMatrix.h"
#include "ProblemInfo.h"

#include <QtCore/QList>
#include <QtCore/QMap>
#include <QHash>

class Network
{
public:
    explicit Network(int);
    explicit Network(const Network *, int);
    virtual ~Network();
    
    /*
     * Apply an input: the two parameters are the input vector (assumed to be of INPUT_SIZE dimension), and
     * the expected class.
     */
    void applyInput(double [], int);
    
    /*
     * Unique identifier for this network.
     */
    int id() const;
    
    /*
     * The last output provided by the network.
     */
    double output() const;
    
    /*
     * The error of the network.
     */
    double error() const;
    
    /*
     * The average error over a test set.
     */
    double averageError() const;
    
    /*
     * The number of links
     */
    int complexity() const;
    
    /*
     * The sparsity computed by the genetic algorithm.
     */
    double sparsity() const;
    
    /*
     * Returns neurons and links in the network, given their IDs; for the Link, we need the IDs of
     * the two neurons connected by it.
     */
    Neuron* getNeuron(int) const;
    Link* getLink(int, int) const;
    
    void setId(int);
    void setAverageError(double);
    void setSparsity(double);
    void addSparsity(double);
    
    /*
     * Performs a mutation on the network.
     */
    void mutate(MutationOperator);
    
    /*
     * The RPROP+ algorithm.
     */
    void updateByRProp();
    
    /*
     * Comparison operator.
     */
    bool operator== (const Network &);
    
private:
    int m_id;
    
    QList< Neuron* > m_inputNeurons;
    QList< Neuron* > m_hiddenNeurons;
    QList< Neuron* > m_outputNeurons;
    QHash< int, Neuron* > m_neurons; /* all the neurons indexed by ID */
    int maxNeuronId; /* the biggest neuron ID currently active (used for mutations) */
    
    LinkMatrix m_connectivity;
    
    double m_lastOutput;
    double m_lastError, m_oldError; /* the "previous" error is used for RPROP+ */
    double m_averageError;
    double m_sparsity;
    
    /*
     * Returns a random weight.
     */
    double randomWeight();
    
    /*
     * Returns a random bias for the new sigmoid neurons.
     */
    double randomBias();
    void createRandomLink(int, int);
    void createBiasLink(Neuron*, Neuron* );
    void computeGradients(int);
    void applyGaussianMutation();
};

#endif