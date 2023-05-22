package com.situalab.dlab;

import java.io.Serializable;
import java.util.List;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;


public class dae_optimization implements Serializable {


    //get dthetaslayer
    public double[][] getgradientthetasAE(double[][] derivativethetasLAE,
                                          double[] aj,
                                          double[] ej){
        for (int row = 0; row < aj.length; row++) {
            for (int col = 0; col < ej.length; col++) {
                derivativethetasLAE[row][col] = aj[row]*ej[col];
            }
        }
        return derivativethetasLAE;
    }

    //derivative Gradient Descent, vanilla & momentum
    public double[][] getderivativeAEBackProp (double[][] thetas,
                                               dae_AccumM GthetaAEAcc,
                                               double lambda,
                                               double mGD){

        double epsilon = 1.e-5;
        double mgd = 1./mGD;
        int row = thetas.length;
        int col = thetas[0].length;
        double[][] Dthetas = new double[row][col];
        double[][] Gthetas = GthetaAEAcc.value();
        //bias gradient, verificado con CS231
        for (int j=0; j<col; j++) {
            Dthetas[0][j] = mgd*Gthetas[0][j];
        }
        //weights gradient, verificado con CS231
        for (int i=1; i<row; i++){
            for (int j=0; j<col; j++){
                Dthetas[i][j] = (mgd*Gthetas[i][j])+(lambda*thetas[i][j]);
            }
        }

        return Dthetas;
    }

    //accumulate gradients, ADAGRAD
    public double[][] accumGradientsAE (double[][] gthetas,
                                        double[][] gthetasAcc){

        double[][] gtA = gthetasAcc; //gradients Acc
        for(int i=0; i<gtA.length; i++) {
            for (int j = 0; j < gtA[0].length; j++) {
                gtA[i][j] += gthetas[i][j]; //accumulate gradients
            }
        }
        return gtA;
    }

    //Vanilla Update
    public double[][] getthetasAEVanillaUpdated(double[][] thetas,
                                                double[][] Dthetas,
                                                double learning,
                                                double[][] DthetasAcc,
                                                double mGD, double lmbda){
        //weight decay update (regularization)
        double weightDecay = 1;
        double[][] thetasW = thetas;

        //adaptive learning
        double ADAlearning = learning / getAlearningLayer(DthetasAcc); //adaptive learning rate

        //bias gradient, nesterov, revisar
        for (int j=0; j<thetas[0].length; j++) {
            thetasW[0][j] -= (ADAlearning)*Dthetas[0][j];
        }
        //weight gradient update
        for(int i=1; i<thetas.length; i++) {
            for (int j = 0; j <thetas[0].length; j++) {

                weightDecay = (1-(ADAlearning*lmbda/mGD)); //(1-(learning*lmbda/n))
                thetasW[i][j] *= weightDecay;
                thetasW[i][j] -= (ADAlearning)*Dthetas[i][j]; //learning/mGD -> StochasticGradientDescent (better without learning/n)
            }
        }
        return thetasW;
    }

    //gradient, l2 norm
    public double getAlearningLayer(double[][] gthetas){
        double ADAlearning = 0.0;
        for (int i = 0; i < gthetas.length; i++) {
            for (int j = 0; j < gthetas[0].length; j++) {
                ADAlearning += pow(gthetas[i][j],2);
            }
        }
        ADAlearning = sqrt(ADAlearning);
        return ADAlearning;
    }

    public List<double[][]> updatethetasAEpretrained(List <double[][]> thetasL,
                                                     double[][] thetas){

        int thetasUpdate = thetasL.size()-1;
        int lastthetasrow = thetasL.get(thetasUpdate).length; //last thetas architecture;
        int lastthetascol = thetasL.get(thetasUpdate)[0].length; //last thetas architecture;

        for (int row=0; row<lastthetasrow; row++){
            for (int col=0; col<lastthetascol; col++){
                thetasL.get(thetasUpdate)[row][col] = thetas[row][col];
            }
        }
        return thetasL;
    }




}
