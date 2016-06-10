#ifndef NEURON_H
#define NEURON_H

#include <QtCore/QList>
#include "Link.h"

using namespace std;

/*
 * Abstract class for a neuron.
 */
class Neuron
{
public:
    enum Layer {
        InputLayer, HiddenLayer, OutputLayer
    };
    
    explicit Neuron(int, Layer);
    explicit Neuron(int, Neuron *); /* create this neuron from another one */
    virtual ~Neuron();
    
    /*
     * Neuron identifier (unique inside the network)
     */
    virtual int id() const;
    
    /*
     * Neuron layer (see enum above)
     */
    virtual Layer layer() const;
    
    /*
     * This is used to compute the gradients.
     */
    virtual double signalError() const;
    
    /*
     * The last output computed by this neuron.
     */
    virtual double output() const;
    
    /*
     * Returns the connections going to or coming from this neuron, as Link classes.
     */
    virtual QList< Link* > outConnections() const;
    virtual QList< Link* > inConnections() const;
    
    /*
     * Manages connections.
     */
    virtual void addInConnection(Link *);
    virtual void removeInConnection(Link *);
    virtual void addOutConnection(Link *);
    virtual void removeOutConnection(Link *);
    virtual void removeOutTowards(Neuron *); /* removes the link going to the specified neuron, if there is one */
    
    virtual void setId(int);
    virtual void setSignalError(double);
    
    /*
     * Each neuron computes the output in its specific way.
     */
    virtual void computeOutput() = 0;
    
    bool operator==(const Neuron &) const;
    
protected:
    int m_id;
    Layer m_layer;
    double m_sigError;
    double m_lastOutput;
    
    QList< Link* > m_inConnections;
    QList< Link* > m_outConnections;
};

class SigmoidNeuron : public Neuron
{
public:
    explicit SigmoidNeuron(int, Neuron::Layer);
    explicit SigmoidNeuron(int, Neuron *);
    virtual ~SigmoidNeuron();
    
    virtual void computeOutput();
};

class TangentNeuron : public Neuron
{
public:
    explicit TangentNeuron(int, Neuron::Layer);
    explicit TangentNeuron(int, Neuron *);
    virtual ~TangentNeuron();

    virtual void computeOutput();
};

#endif