#pragma once

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include <set>

class EventManager
{
public:
    enum State
    {
        INITIALIZED,
        PAUSED,
        RUNNING,
        FINISHED,
        NEW_FRAME,
        CONFIGURE,
        QUIT
    };

    enum Event
    {
        NONE,
        RUN, //ok
        STEP, //ok
        PAUSE,
        RESET,
        GRAB_FRAME,
        CONTINUE,
        UPDATE_RENDERCONTROLS,
        ADD_SEQUENCE,
        START_RECORDING,
        END_RECORDING,
        SAVE_CONFIG,
        TOGGLE_HELP, //ok
        NEXT_OBJECT, //ok
        PREVIOUS_OBJECT, //ok
        UPDATE_RENDERER,
        UPDATE_IMAGE,
        NEXT_SEQUENCE, //ok
        PREVIOUS_SEQUENCE, //ok
        CLOSE //ok
    };
private:
    boost::condition_variable cnd_;
    boost::mutex m_mutex;
    Event e;
protected:
    std::set<Event> availableEvents;
public:

    EventManager()
    {
        e = NONE;

        availableEvents.insert(TOGGLE_HELP);
    }

    virtual void pipelineStateChanged(State state, std::string activeStepName="") = 0;

    bool isEventAvailable(Event e)
    {
        availableEvents.count(e) > 0;
    }

    void removeAvailableEvent(Event e)
    {
        std::cout << "remove available event " << e << std::endl;
        availableEvents.erase(e);
    }

    void addAvailableEvent(Event e)
    {
        availableEvents.insert(e);
    }

    virtual bool onEvent(Event e)
    {
        return false;
    }

    void trigger(Event e)
    {
        if (!isEventAvailable(e))
        {
            std::cout << "Event " << e << " not available" << std::endl;
            return;
        }

        bool handled = onEvent(e);

        if (!handled)
        {
            std::cout << "event forwarded to consumer" << std::endl;

            {
                boost::mutex::scoped_lock lock(m_mutex);
                this->e = e;
                cnd_.notify_one();
            }
        } else {
            std::cout << "event" << e << "handled by renderer" << std::endl;
        }
    }

    Event consumeNonBlocking()
    {
        boost::mutex::scoped_lock lock(m_mutex);

        Event result = e;
        e = NONE;
        return result;
    }

    Event consumeBlocking()
    {
        boost::mutex::scoped_lock lock(m_mutex);
        while (e == NONE) cnd_.wait(lock);

        Event result = e;
        e = NONE;
        return result;
    }
};
