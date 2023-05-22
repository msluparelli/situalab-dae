package com.situalab.dlab;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.util.DoubleAccumulator;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.log;


public class dae_feedback_regular_bug implements Serializable {

    public void getAutoencoders(LabeledPoint xvalues,
                                List<double[][]> thetasL,
                                List<double[][]> thetasTL,
                                dae_AccumM GthetaAEAcc,
                                DoubleAccumulator JthetasCost,
                                DoubleAccumulator mAE,
                                String AEmethod) throws IOException {

        dae_thetas AEthetas = new dae_thetas();

        //feature init
        double[] biasUnit = {1}; double[] zi = new double[1]; double[] axhat; double[] xhat;
        double[] xinput = new double[1]; //init xinput, used to get gradient (xhat-xinput)
        int lastthetas = thetasL.size()-1; //last thetas to backpropagate
        List<double[]> A = new ArrayList<double[]>();


        //x values
        double[] xvalue = xvalues.features().toArray(); //x values


        //feedforward, hiddenLayer
        double[] ai = ArrayUtils.addAll(biasUnit, xvalue); //xinput
        double[] aitilde = ai; //used for denoising/constrain AE
        for (int hiddenLayer=0; hiddenLayer<thetasL.size(); hiddenLayer++){
            xinput = ai; //fix input value to get xhat-xinput of the last hiddenlayer
            aitilde = ai; //fix aitilde
            aitilde = getvaluestilde(ai, aitilde, AEmethod); //regular/denoising/constrain AE
            zi = multiMV(thetasTL.get(hiddenLayer), aitilde); //zetas from regular/denoising/constrain AE
            ai = Arrays.stream(zi).map(xoz -> dae_activation.sigmoid.apply(xoz)).toArray(); //encoder, sigmoid function
            ai = ArrayUtils.addAll(biasUnit, ai); //add bias for next hiddenLayer
        }


        //hidden layer decoder, from now on only for the last thetas
        double[][] thetasStar = AEthetas.getthetasStar(thetasL.get(lastthetas)); //only last thetas
        axhat = multiMV(thetasStar, ai); //last ai from last hiddenLayer trained
        xhat = Arrays.stream(axhat).map(xoz -> dae_activation.sigmoid.apply(xoz)).toArray(); //decoder, sigmoid function


        //get gradient & cost function
        double[] ejxhat = new double[xhat.length];double loglCost = 0;
        for (int g=0; g<xhat.length; g++){
            ejxhat[g] = xhat[g]-xinput[g]; //gradient, OJO!! bias incluido, xinput is preceding layer
            if (g!=0) {
                loglCost += -(xinput[g]*log(xhat[g]) + (1-xinput[g])*log(1-xhat[g])); //cost, OJO!! bias excluido
            }

        }
        JthetasCost.add(loglCost);

        getBackPropagationAE(ejxhat, zi, thetasStar, aitilde, GthetaAEAcc);

        mAE.add(1.0);

    }


    //backpropagation algorithm
    public void getBackPropagationAE(double[] ej,
                                     double[] zi,
                                     double[][] thetasStar,
                                     double[] xi,
                                     dae_AccumM GthetaAEAcc){

        dae_thetas AEthetas = new dae_thetas();

        //dzi
        double[] biasUnit = {1}; //bias unit
        double[] dziBPAE = Arrays.stream(zi).map(xoz -> dae_activation.dsigmoid.apply(xoz)).toArray(); //get dactivation z
        dziBPAE = ArrayUtils.addAll(biasUnit, dziBPAE); //deactivated ax, incluye bias unit

        //thetasStartT & ej hidden layer init
        double[][] thetasStarT = AEthetas.getTransposeAE(thetasStar);
        double[] ejencoder = new double[(thetasStar[0].length)]; //ej hidden layer

        //get error hidden layer
        getEjAE(ejencoder, thetasStarT, ej, dziBPAE); //get ejl{thetasStar, ejdecoder, z(ax)}

        //get derivative
        ej = Arrays.copyOfRange(ejencoder,1, ejencoder.length); // ej(l), remove bias unit
        double[][] gthetahiddenlayer = new double[xi.length][ej.length]; //init gradient matrix
        getgradientthetasAE(gthetahiddenlayer, xi, ej); //get gradient
        GthetaAEAcc.add(gthetahiddenlayer); //gradient accumulator works fine

    }

    public double[] getvaluestilde (double[] xvalue,
                                    double[] xtilde,
                                    String AEmethod){

        if (AEmethod.equals("denoising")) {

            Random rand = new Random();
            rand.setSeed(76835279);
            rand.longs(xvalue.length);
            for (int r=1; r<xvalue.length; r++){
                double rindx = 0;
                double rvalue = rand.nextDouble();
                if (rvalue>=0.5) {rindx = 1;}
                xtilde[r] = xtilde[r]*rindx; //denoising autoencoder
            }
        }

        return xtilde;
    }



    //get zi values (matrix, vector multplication)
    public double[] multiMV(double[][] M, double[] v){
        double[] Mv = new double[M.length];
        for (int row=0; row<M.length; row++){
            double rowSum = 0;
            for (int col=0; col<M[row].length; col++){
                rowSum += (M[row][col]*v[col]);
            }
            Mv[row] = rowSum;
        }
        return Mv;
    }




    //get ej
    public void getEjAE(double[] ejout, double[][] Wij, double[] ej, double[] dzi) {
        for (int row = 0; row < Wij.length; row++) {
            double rowSum = 0;
            for (int col = 0; col < Wij[row].length; col++) {
                rowSum += (Wij[row][col]*ej[col]);
            }
            ejout[row] = rowSum*dzi[row];
        }
    }

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


}
