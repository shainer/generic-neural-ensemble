#include "Network.h"
#include "Utils.h"
#include "ProblemInfo.h"
#include <stdlib.h>

#include <QtCore/QPair>
#include <QtCore/QDebug>
#include <iostream>
#include <cmath>

Network::Network(int id)
    : m_id(id)
    , maxNeuronId(0)
    , m_averageError(0.0)
    , m_sparsity(0.0)
{
    int numNeurons = INPUT_SIZE + OUTPUT_SIZE + HIDDEN_SIZE;
    
    /*
     * This is a fake neuron to represent the predecessor neuron for biases link
     * (see below). The layer is meaningless, while the ID cannot be taken by any
     * other neuron in the network.
     */
    Neuron* dummy = new SigmoidNeuron(-1, Neuron::InputLayer);
    m_neurons.insert(-1, dummy);
    
    for (int i = 1; i <= numNeurons; i++) {
        Neuron::Layer layer = Neuron::HiddenLayer;
        
        if (i <= INPUT_SIZE) {
            Neuron* neuron = new SigmoidNeuron(i, Neuron::InputLayer);
            
            /*
             * This is a fake connection, always with weight 1, used to set the input attribute to this neuron.
             */
            neuron->addInConnection( new Link(1.0f, NULL, neuron) );
            
            /*
             * Adds a fake link to represent biases. This is easier than setting biases in the neurons,
             * because we can mutate and train them as we do with weights.
             * 
             * The output is set now to 1.0 so the neuron will get the bias as input, and will never be
             * changed.
             */
            createBiasLink(dummy, neuron);
            
            m_inputNeurons.append(neuron);
            m_neurons.insert(i, neuron);
            
            continue;
        }
        
        Neuron* neuron = NULL;
        
        /*
         * Please note the output neurons have IDs smaller than the hidden neurons'. This is because we may need to add/remove
         * hidden neurons later, due to mutations, while the input and output neurons never change.
         */
        if (i <= INPUT_SIZE + OUTPUT_SIZE) {
            layer = Neuron::OutputLayer;
            neuron = new TangentNeuron(i, layer);
        } else {
            neuron = new SigmoidNeuron(i, layer);
        }
        
        createBiasLink(dummy, neuron);
        m_neurons.insert(i, neuron);
        
        if (layer == Neuron::OutputLayer) {
            m_outputNeurons.append(neuron);
        } else {
            m_hiddenNeurons.append(neuron);
        }
    }
    
    maxNeuronId = numNeurons;
    
    /*
     * No connections between hidden neurons are allowed
     */
    for (int i = 1; i <= INPUT_SIZE; i++) {
        for (int j = (INPUT_SIZE + OUTPUT_SIZE + 1); j <= numNeurons; j++) {
            createRandomLink(i, j);
        }
    }
    
    for (int i = INPUT_SIZE + OUTPUT_SIZE + 1; i <= numNeurons; i++) {
        for (int j = (INPUT_SIZE + 1); j <= INPUT_SIZE + OUTPUT_SIZE; j++) {
            createRandomLink(i, j);
        }
    }
}

/*
 * Makes an hard copy of this network. Used during breeding for the genetic algorithm.
 */
Network::Network(const Network* other, int id)
    : m_id(id)
    , maxNeuronId(0)
    , m_sparsity(0)
{    
    Q_FOREACH (Neuron* otherNeuron, other->m_neurons.values()) {
        int neuronId = otherNeuron->id();
        
        if (neuronId == -1) {
            m_neurons.insert(-1, new SigmoidNeuron(-1, Neuron::InputLayer));
            continue;
        }
        
        switch ( otherNeuron->layer() ) {
            case Neuron::InputLayer: {
                Neuron* newNeuron = new SigmoidNeuron(neuronId, otherNeuron);
                newNeuron->addInConnection( new Link( 1.0f, 0, newNeuron) );
                
                m_inputNeurons.append(newNeuron);
                m_neurons.insert(neuronId, newNeuron);
                break;
            }
            
            case Neuron::HiddenLayer: {
                Neuron* newNeuron = new SigmoidNeuron(neuronId, otherNeuron);
                
                m_neurons.insert(neuronId, newNeuron);
                m_hiddenNeurons.append(newNeuron);
                break;
            }
            
            case Neuron::OutputLayer: {
                Neuron* newNeuron = new TangentNeuron(neuronId, otherNeuron);
                
                m_neurons.insert(neuronId, newNeuron);
                m_outputNeurons.append(newNeuron);
                break;
            }
        }
    }
    
    maxNeuronId = other->maxNeuronId;
    
    Q_FOREACH (Link* link, other->m_connectivity.links()) {
        int in = link->predecessor()->id();
        int out = link->successor()->id();
        
        Link* newLink = new Link(link->weight(), m_neurons[in], m_neurons[out]);
        newLink->setOutput( link->output() );

        m_neurons[in]->addOutConnection(newLink);
        m_neurons[out]->addInConnection(newLink);
        m_connectivity.addLink(in, out, newLink);
    }
}

Network::~Network()
{    
    qDeleteAll(m_neurons);
    m_neurons.clear();
}

/*
 * Creates a link between i and j with 50% probability.
 */
void Network::createRandomLink(int i, int j)
{
    double r = randomDouble(0.0, 1.0);
    
    if (r <= 0.5) {
        Link* link = new Link(randomWeight(), m_neurons[i], m_neurons[j]);
        m_neurons[i]->addOutConnection(link);
        m_neurons[j]->addInConnection(link);
        
        m_connectivity.addLink(i, j, link);
    }
}

/*
 * Creates a fake link to represent a bias
 */
void Network::createBiasLink(Neuron* dummy, Neuron* neuron)
{
    Link* biasLink = new Link(randomBias(), dummy, neuron);
    biasLink->setOutput(1.0);
    neuron->addInConnection(biasLink);
    m_connectivity.addLink(-1, neuron->id(), biasLink);
}

double Network::randomBias()
{
    return randomDouble(-6.0, 1.0);
}

double Network::randomWeight()
{
    return randomDouble(-0.2, 0.2);
}

Neuron* Network::getNeuron(int id) const
{
    return m_neurons[id];
}

Link* Network::getLink(int n1, int n2) const
{
    return m_connectivity.link(n1, n2);
}

bool Network::operator==(const Network& other)
{
    return m_id == other.id();
}

void Network::setId(int i)
{
    m_id = i;
}

int Network::id() const
{
    return m_id;
}

void Network::applyInput(double input[], int expectedClass)
{
    for (int i = 0; i < m_inputNeurons.size(); i++) {        
        Neuron* neuron = m_inputNeurons[i];
        Link* inLink = neuron->inConnections().first();
        inLink->setOutput( input[i] );
        
        neuron->computeOutput();
    }
    
    Q_FOREACH (Neuron* hidden, m_hiddenNeurons) {
        hidden->computeOutput();
    }
    
    Neuron* outNeuron = m_outputNeurons.first();
    outNeuron->computeOutput();
    m_lastOutput = (outNeuron->output() > 0.0) ? 1.0 : 0.0;
    
    m_oldError = m_lastError;
    m_lastError = (expectedClass == m_lastOutput) ? 0.0 : 1.0; /* simple classification error */
    
    computeGradients(expectedClass);
}

void Network::computeGradients(int expectedClass)
{
    double target = (expectedClass == 0) ? -1 : 1;
    
    double out = m_outputNeurons.first()->output();
    double oGradient = (1 - out) * out * (target - out);

    Q_FOREACH (Link* inLink, m_outputNeurons.first()->inConnections()) {
        inLink->setGradient(oGradient);
    }
    
    for (int i = 0; i < m_hiddenNeurons.size(); i++) {
        Neuron* hidden = m_hiddenNeurons[i];
        
        double out = hidden->output();
        double derivative = (1 - out) * out;
        double sum = 0.0;
        
        Q_FOREACH (Link* outGoing, hidden->outConnections()) {
            sum += oGradient * outGoing->weight();
        }
        
        Q_FOREACH (Link* inLink, hidden->inConnections()) {
            inLink->setGradient(derivative * sum);
        }
        
        return;
    }
}

void Network::mutate(MutationOperator op)
{
    switch (op) {

        case RemoveLink: {
            if (m_connectivity.complexity() < LINK_SIZE_MIN) { /* avoid removing too many links, and mutate the weights instead */
                applyGaussianMutation();
                break;
            }
            
            int in = 0;
            int out = 0;
            
            int linkType = randomInteger(1, 2);
            int inMin, inMax, outMin, outMax;
            inMin = inMax = outMin = outMax = 0;
            
            /*
             * Determines in which range the IDs should be randomly chosen, depending on whether we're removing a link
             * between input and hidden, or between hidden and output (no other kind is allowed).
             */
            switch (linkType) {
                case 1: { /* Link between input and hidden neuron */
                    inMin = 1;
                    inMax = INPUT_SIZE;
                    outMin = INPUT_SIZE + OUTPUT_SIZE + 1;
                    outMax = maxNeuronId;
                    break;
                }
                
                case 2: { /* Between hidden and output */
                    inMin = INPUT_SIZE + OUTPUT_SIZE + 1;
                    inMax = maxNeuronId;
                    outMin = INPUT_SIZE + 1;
                    outMax = INPUT_SIZE + OUTPUT_SIZE;
                    break;
                }
                
                default:
                    break;
            }
            
            int attempt = 0;
            
            do {
                in = randomInteger(inMin, inMax);
                out = randomInteger(outMin, outMax);
                
                if (++attempt > 20) {
                    return;
                }
            } while (!m_connectivity.link(in, out));
            
            Link* link = m_connectivity.link(in, out);
            link->predecessor()->removeOutConnection(link);
            link->successor()->removeInConnection(link);
            
            m_connectivity.removeLink(in, out);
            delete link;
            link = 0;
            break;
        }
        
        case AddLink: {
            int in = 0;
            int out = 0;
            bool found = false;
            
            int linkType = randomInteger(1, 2);
            int inMin, inMax, outMin, outMax;
            inMin = inMax = outMin = outMax = 0;
            
            /*
             * Again determines the accepted range (see RemoveLink case).
             */
            switch (linkType) {
                case 1: { /* Link between input and hidden neuron */
                    inMin = 1;
                    inMax = INPUT_SIZE;
                    outMin = INPUT_SIZE + OUTPUT_SIZE + 1;
                    outMax = maxNeuronId;
                    break;
                }
                
                case 2: { /* Link between hidden and output neuron */
                    inMin = INPUT_SIZE + OUTPUT_SIZE + 1;
                    inMax = maxNeuronId;
                    outMin = INPUT_SIZE + 1;
                    outMax = INPUT_SIZE + OUTPUT_SIZE;
                    break;
                }
            }
            
            /*
             * Check all the pairs of neurons, and pick the first pair that isn't connected by a link.
             */
            for (in = inMin; in <= inMax; in++) {
                for (out = outMin; out <= outMax; out++) {
                    if (!m_connectivity.link(in, out) && !m_connectivity.link(out, in)) {
                        found = true;
                        
                        Link* link = new Link(randomWeight(), m_neurons[in], m_neurons[out]);
                        link->predecessor()->addOutConnection(link);
                        link->successor()->addInConnection(link);
                        
                        m_connectivity.addLink(in, out, link);
                        break;
                    }
                }
                
                if (found) {
                    break;
                }
            }
            break;
        }
        
        /*
         * NOTE: when removing a neuron, the last one in the list, i.e. the one with the biggest ID, gets the ID of the removed
         * one (if they don't coincide). This simplifies a lot of loops in the network.
         */
        case RemoveNeuron: {
            if (m_hiddenNeurons.size() <= HIDDEN_SIZE_MIN) { /* avoid removing all neurons, and mutate weights instead */
                applyGaussianMutation();
                break;
            }
            
            int neuronId = randomInteger(INPUT_SIZE + OUTPUT_SIZE + 1, maxNeuronId);
            Neuron* neuron = m_neurons[neuronId];
            
            m_hiddenNeurons.removeOne(neuron);
            m_neurons.remove(neuronId);
            m_connectivity.removeAllLinks(neuronId, maxNeuronId, m_neurons);
            
            if (neuronId != maxNeuronId) {
                Neuron* latestNeuron = m_neurons[maxNeuronId];
                int oldId = latestNeuron->id();
                latestNeuron->setId(neuronId);
                
                m_neurons.remove(oldId);
                m_neurons.insert(neuronId, latestNeuron);
                m_connectivity.changeId(oldId, neuronId);
            }
            
            maxNeuronId--;
            delete neuron;
            neuron = 0;
            break;
        }
        
        /*
         * We always add an hidden neuron.
         */
        case AddNeuron: {
            if (m_hiddenNeurons.size() >= HIDDEN_SIZE_MAX) {
                applyGaussianMutation();
                break;
            }
            
            int neuronId = ++maxNeuronId;
            Neuron* neuron = new SigmoidNeuron(neuronId, Neuron::HiddenLayer);
            
            createBiasLink(m_neurons[-1], neuron);
            m_neurons.insert(neuronId, neuron);
            m_hiddenNeurons.append(neuron);
            
            /*
             * Randomly connects with the input layer
             */
            Q_FOREACH (Neuron* inputNeuron, m_inputNeurons) {
                int choice = randomInteger(1, 2);
                
                if (choice == 1) {
                    Link* link = new Link(randomWeight(), inputNeuron, neuron);
                    inputNeuron->addOutConnection(link);
                    neuron->addInConnection(link);
                    
                    m_connectivity.addLink(inputNeuron->id(), neuronId, link);
                }
            }
            
            /*
             * Since this is a binary classification problem, there is only one output neuron, and we always connect this one
             * to it (the link may be removed by further mutations).
             */
            Q_FOREACH (Neuron* outputNeuron, m_outputNeurons) {
                Link* link = new Link(randomWeight(), neuron, outputNeuron);
                neuron->addOutConnection(link);
                outputNeuron->addInConnection(link);
                
                m_connectivity.addLink(neuronId, outputNeuron->id(), link);
            }
            
            break;
        }
        
        case WeightMutation: {
            applyGaussianMutation();
            break;
        }
        
        default:
            break;
    }
}

void Network::applyGaussianMutation()
{
    Q_FOREACH (Link* link, m_connectivity.links()) {
        double newWeight = gaussianMutation( link->weight(), 0, 0.05 );
        link->setWeight(newWeight + link->weight());
    }
}

double Network::error() const
{
    return m_lastError;
}

double Network::output() const
{
    return m_lastOutput;
}

int Network::complexity() const
{
    return m_connectivity.complexity();
}

double Network::sparsity() const
{
    return m_sparsity;
}

void Network::setSparsity(double s)
{
    m_sparsity = s;
}

void Network::addSparsity(double s)
{
    m_sparsity += s;
}

void Network::updateByRProp()
{    
    /* Train weights */
    Q_FOREACH (Link* link, m_connectivity.links()) {        
        double gradient = link->gradient();
        double signChange = gradient * link->previousGradient();
        
//         qDebug() << "Gradients:" << gradient << link->previousGradient();
        
        double delta = 0.0f;
        double oldDelta = link->delta();
        double weightChange = 0.0;
        
        if (signChange > 0) {
            delta = minimum(oldDelta * POSITIVE_ETA, MAX_STEP);
            weightChange = (gradient > 0) ? -delta : +delta; /* sign function */
        }
        else if (signChange < 0) {
            delta = max(oldDelta * NEGATIVE_ETA, MIN_STEP);
            
            if (m_lastError > m_oldError) { /* Rprop+ condition */
                weightChange = (gradient > 0) ? +delta : -delta;
            }
            link->setGradient(0);
        } else {
            delta = oldDelta;
            weightChange = (gradient > 0) ? -oldDelta : +oldDelta;
        }

        link->setWeight( link->weight() + weightChange );
        link->setDelta(delta);
    }
}

double Network::averageError() const
{
    return m_averageError;
}

void Network::setAverageError(double a)
{
    m_averageError = a;
}