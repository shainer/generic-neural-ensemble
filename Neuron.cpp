#include "Neuron.h"
#include "Utils.h"
#include <cmath>
#include <iostream>

using namespace std;

Neuron::Neuron(int id, Neuron::Layer layer)
    : m_id(id)
    , m_layer(layer)
    , m_sigError(0.0f)
{}

Neuron::Neuron(int id, Neuron* other)
    : m_id(id)
    , m_layer( other->layer() )
    , m_sigError(0.0f)
{}

Neuron::~Neuron()
{}

int Neuron::id() const
{
    return m_id;
}

void Neuron::setId(int id)
{
    m_id = id;
}

Neuron::Layer Neuron::layer() const
{
    return m_layer;
}

double Neuron::output() const
{
    return m_lastOutput;
}

double Neuron::signalError() const
{
    return m_sigError;
}

QList< Link* > Neuron::inConnections() const
{
    return m_inConnections;
}

QList< Link* > Neuron::outConnections() const
{
    return m_outConnections;
}

void Neuron::addInConnection(Link* link)
{
    m_inConnections.push_front(link);
}

void Neuron::removeInConnection(Link* link)
{
    m_inConnections.removeOne(link);
}

void Neuron::addOutConnection(Link* link)
{
    m_outConnections.push_front(link);
}

void Neuron::removeOutConnection(Link* link)
{
    m_outConnections.removeOne(link);
}

void Neuron::removeOutTowards(Neuron* next)
{
    for (QList< Link* >::iterator it = m_outConnections.begin(); it != m_outConnections.end(); it++) {
        Link* link = (*it);
        
        if (*(link->successor()) == (*next)) {
            m_outConnections.erase(it);
            delete link;
        }
    }
}

void Neuron::setSignalError(double err)
{
    m_sigError = err;
}

bool Neuron::operator==(const Neuron& other) const
{
    return m_id == other.id();
}

SigmoidNeuron::SigmoidNeuron(int id, Neuron::Layer layer)
    : Neuron(id, layer)
{}

SigmoidNeuron::SigmoidNeuron(int id, Neuron* other)
    : Neuron(id, other->layer())
{}

SigmoidNeuron::~SigmoidNeuron()
{}

void SigmoidNeuron::computeOutput()
{
    double z = 0.0;
    
    /*
     * Takes the output of the predecessor neuron and the weight of the link,
     * for each incoming connection.
     */
    Q_FOREACH (Link* in, m_inConnections) {
        z += (in->output() * in->weight());
    }
    
    m_lastOutput = 1.0f / (1.0f + exp(-z));
    
    /*
     * Sets the output on the outgoing link, so it can be retrieved by the
     * successor neuron.
     */
    Q_FOREACH (Link* out, m_outConnections) {
        out->setOutput(m_lastOutput);
    }
}

TangentNeuron::TangentNeuron(int id, Neuron::Layer layer)
    : Neuron(id, layer)
{}

TangentNeuron::TangentNeuron(int id, Neuron* other)
    : Neuron(id, other->layer())
{}

TangentNeuron::~TangentNeuron()
{}

void TangentNeuron::computeOutput()
{
    double z = 0.0;
    
    Q_FOREACH (Link* in, m_inConnections) {        
        z += (in->output() * in->weight());
    }
    
    if (z < -10.0) {
        m_lastOutput = -1.0;
    } else if (z > 10.0) {
        m_lastOutput = 1.0;
    } else {
        m_lastOutput = tanh(z);
    }
        
    Q_FOREACH (Link* out, m_outConnections) {
        out->setOutput(m_lastOutput);
    }    
}