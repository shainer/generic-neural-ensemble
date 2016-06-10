/*
 * This class defines a link between two neurons, called predecessor and successor
 */

#include "Link.h"
#include "Neuron.h"
#include "ProblemInfo.h"

Link::Link(double weight, Neuron* prev, Neuron* succ)
    : m_weight(weight)
    , m_output(0.0)
    , m_gradient(0.0)
    , m_prevGradient(0.0)
    , m_delta(INITIAL_STEP)
    , m_next(succ)
    , m_prev(prev)
{}

Link::~Link()
{}

double Link::weight() const
{
    return m_weight;
}

double Link::output() const
{
    return m_output;
}

double Link::gradient() const
{
    return m_gradient;
}

double Link::previousGradient() const
{
    return m_prevGradient;
}

double Link::delta() const
{
    return m_delta;
}

Neuron* Link::predecessor() const
{
    return m_prev;
}

Neuron* Link::successor() const
{
    return m_next;
}

void Link::setWeight(double w)
{
    m_weight = w;
}

void Link::setOutput(double o)
{
    m_output = o;
}

void Link::setGradient(double g)
{
    m_prevGradient = m_gradient;
    m_gradient = g;
}

void Link::setDelta(double d)
{
    m_delta = d;
}
