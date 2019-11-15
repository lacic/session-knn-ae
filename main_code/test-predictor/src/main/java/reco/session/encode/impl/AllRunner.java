package reco.session.encode.impl;

import java.util.ArrayList;

/**
 * Composite pattern for running different Runners in an ordered way 
 *
 */
public class AllRunner implements Runnable {

    private ArrayList<Runnable> runners;

    /**
     * @param runners ordered list of runners to run
     */
    public AllRunner(ArrayList<Runnable> runners) {
        this.runners = runners;
    }

    @Override
    public void run() {
        for (Runnable runner : runners) {
            runner.run();
        }
    }


}
