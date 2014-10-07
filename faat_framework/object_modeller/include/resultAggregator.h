#pragma once

#include <vector>

#include "result.h"

#include <boost/tuple/tuple.hpp>

namespace object_modeller
{

template<class TResult, class TupleType, size_t N>
class Transform
{
private:
    TResult *result;
    int sequenceId;
public:
    Transform(TResult *result, int sequenceId)
    {
        this->result = result;
        this->sequenceId = sequenceId;
    }

   void call(TupleType* t)
   {
       boost::tuples::get<N>(*result) = boost::tuples::get<N>(*t).getResult(sequenceId);
       Transform<TResult, TupleType, N-1> next(result, sequenceId);
       next.call(t);
   }
};

template<class TResult, class TupleType>
class Transform<TResult, TupleType, 0>
{
private:
    TResult *result;
    int sequenceId;
public:
    Transform(TResult *result, int sequenceId)
    {
        this->result = result;
        this->sequenceId = sequenceId;
    }

   void call(TupleType* t)
   {
       boost::tuples::get<0>(*result) = boost::tuples::get<0>(*t).getResult(sequenceId);
   }
};


template<class T, class U>
class ResultAggregator : public ResultBase<T>
{
private:
    U *input;
public:
    ResultAggregator(U *input) // U is eg. tuple<Result<string>, Result<int> >
    {
        this->input = input;
    }

    T transform(U* t, int sequenceId)
    {
       T result;
       Transform<T, U, boost::tuples::length<U>::value-1> t2(&result, sequenceId);
       t2.call(t);
       return result;
    }

    T getResult(int sequenceId) // T is eg. tuple<string, int>
    {
        return transform(input, sequenceId);
    }

    virtual int getNrSequences()
    {
        return boost::tuples::get<0>(*input).getNrSequences();
    }

};


}
