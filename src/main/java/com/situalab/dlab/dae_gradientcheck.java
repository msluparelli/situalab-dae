package com.situalab.dlab;

import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.util.DoubleAccumulator;

import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.DoubleStream;

import static java.lang.Math.log;


public class dae_gradientcheck {

    public void checkGradient(LabeledPoint event,
                              String AEmethod,
                              Map<String,String> aeparams) throws IOException {

        //hyperparams
        double lmbda = 1.e-10; //regularisation term
        double epsilon = 10.e-4;

        //classes
        dae_thetas AEthetas = new dae_thetas();
        dae_feedback_regular AEfeedbackregular = new dae_feedback_regular();
        //dae_feedback_regular_bug AEfeedback = new dae_feedback_regular_bug();
        dae_feedback_sparse AEfeedbacksparse = new dae_feedback_sparse();


        //AE params, regular
        int row = event.features().size();int col = 4;
        List<double[][]> thetasL = new ArrayList<double[][]>(); //thetas List
        thetasL.add(AEthetas.getThetasAE(row,col));
        List<double[][]> thetasTL = AEthetas.getthetasTList(thetasL); //thetas transpose
        int thetasrow = thetasL.get(0).length; //last thetas architecture
        int thetascol = thetasL.get(0)[0].length; //last thetas architecture
        int lasthetas = thetasL.size()-1; //last thetas to update
        double[][] thetasS = AEthetas.getthetasS(thetasL.get(lasthetas)); //only last thetas
        int thetasSrow = thetasS.length;
        int thetasScol = thetasS[0].length;
        double[][] thetasST = AEthetas.getTransposeAE(thetasS); //thetas Star Transpose
        dae_AccumM GthetaAEAcc = new dae_AccumM(thetasrow, thetascol);
        dae_AccumM GthetaSAEAcc = new dae_AccumM(thetasSrow, thetasScol);
        DoubleAccumulator Jtheta = new DoubleAccumulator(); //J(theta) Accumulator
        DoubleAccumulator m = new DoubleAccumulator(); //J(theta) Accumulator
        DoubleAccumulator sparsityValue = new DoubleAccumulator(); //J(theta) Accumulator
        dae_AccumV pjAESparseAcc = new dae_AccumV(thetascol); //include bias

        //AE params, sparse
        double[] pj = new double[thetascol]; //include bias
        double[] KLg = new double[thetascol]; //include bias
        double p = 0.05; //sparsity
        double beta = 1.e-4; //sparsity regularisation penalty


        //AEfeedback
        if(AEmethod.equals("regular") | AEmethod.equals("denoising")){
            AEfeedbackregular.getAutoencoders(event, thetasL, thetasTL, thetasS, thetasST, GthetaAEAcc, GthetaSAEAcc, Jtheta, m, AEmethod); //0.30257
            //AEfeedback.getAutoencoders(event, thetasL, thetasTL, GthetaAEAcc, Jtheta, m, AEmethod); //0.37155
            System.out.print("AEmethodCG:"+AEmethod);
        }
        if(AEmethod.equals("sparse")){
            getAEsparse(event, thetasL, thetasTL, thetasS, thetasST, GthetaAEAcc, GthetaSAEAcc, Jtheta, m, sparsityValue, pj, p, KLg, beta, pjAESparseAcc);
            System.out.print("AEmethodCG:"+AEmethod);
        }
        double[] gthetasAEAccflat = Arrays.stream(GthetaAEAcc.value()).flatMapToDouble(Arrays::stream).toArray();


        double[][] gAEaprox = new double[thetasrow][thetascol];
        for (int i=0; i<thetasL.get(0).length; i++){
            for (int j=0; j<thetasL.get(0)[0].length; j++) {

                //thetaPlus
                List<double[][]> thetasLplus = new ArrayList<double[][]>(); //thetas List
                thetasLplus.add(AEthetas.getThetasAE(row,col));
                thetasLplus.get(0)[i][j] += epsilon; //10.e-4
                List<double[][]> thetasTLplus = AEthetas.getthetasTList(thetasLplus); //thetas transpose
                DoubleAccumulator Jthetaplus = new DoubleAccumulator(); //J(theta) Accumulator
                DoubleAccumulator mplus = new DoubleAccumulator(); //J(theta) Accumulator

                if(AEmethod.equals("regular") | AEmethod.equals("denoising")){
                    AEfeedbackregular.getAutoencoders(event, thetasLplus, thetasTLplus, thetasS, thetasST, GthetaAEAcc, GthetaSAEAcc, Jthetaplus, mplus, AEmethod);
                    //AEfeedback.getAutoencoders(event, thetasLplus, thetasTLplus, GthetaAEAcc, Jthetaplus, mplus, AEmethod);
                }
                if(AEmethod.equals("sparse")){
                    getAEsparse(event, thetasLplus, thetasTLplus, thetasS, thetasST, GthetaAEAcc, GthetaSAEAcc, Jthetaplus, mplus, sparsityValue, pj, p, KLg, beta, pjAESparseAcc);
                }

                double sqwtplus = AEthetas.getthetassqw(thetasLplus.get(0)); //thetas squared
                double costplus = (1./mplus.value() * Jthetaplus.value()) + (lmbda/2*sqwtplus); //cost reg


                //thetaMinus
                List<double[][]> thetasLmin = new ArrayList<double[][]>(); //thetas List
                thetasLmin.add(AEthetas.getThetasAE(row,col));
                thetasLmin.get(0)[i][j] -= epsilon;
                List<double[][]> thetasTLmin = AEthetas.getthetasTList(thetasLmin); //thetas transpose
                DoubleAccumulator Jthetamin = new DoubleAccumulator(); //J(theta) Accumulator
                DoubleAccumulator mmin = new DoubleAccumulator(); //J(theta) Accumulator

                if(AEmethod.equals("regular") | AEmethod.equals("denoising")){
                    AEfeedbackregular.getAutoencoders(event, thetasLmin, thetasTLmin, thetasS, thetasST, GthetaAEAcc, GthetaSAEAcc, Jthetamin, mmin, AEmethod);
                    //AEfeedback.getAutoencoders(event, thetasLmin, thetasTLmin, GthetaAEAcc, Jthetamin, mmin, AEmethod);
                }
                if(AEmethod.equals("sparse")){
                    getAEsparse(event, thetasLmin, thetasTLmin, thetasS, thetasST, GthetaAEAcc, GthetaSAEAcc, Jthetamin, mmin, sparsityValue, pj, p, KLg, beta, pjAESparseAcc);
                }

                double sqwtmin = AEthetas.getthetassqw(thetasLmin.get(0)); //thetas squared
                double costmin = (1./mmin.value() * Jthetamin.value()) + (lmbda/2*sqwtmin); //cost reg

                //gradient check
                gAEaprox[i][j] = (costplus-costmin) / (2*epsilon);

            }

        }

        double[] gAEaproxV = Arrays.stream(gAEaprox).flatMapToDouble(Arrays::stream).toArray();


        double gradcheck = DoubleStream.of(gAEaproxV).sum();
        double gthetasbp = DoubleStream.of(gthetasAEAccflat).sum();
        double gradientc = gradcheck-gthetasbp;

        NumberFormat formatter = new DecimalFormat("#0.00000");
        String gradientbackp = formatter.format(gthetasbp);
        String gradientcheck = formatter.format(gradcheck);
        String gradientdiffe = formatter.format(gradientc);



        //System.out.println("\nGradient Checking");
        System.out.print(" gradientback:"+gradientbackp+" gradientcheck:"+gradientcheck+" difference:"+gradientdiffe+"\n");



    }

    public void getAEsparse(LabeledPoint event,
                            List<double[][]> thetasL,
                            List<double[][]> thetasTL,
                            double[][] thetasS,
                            double[][] thetasST,
                            dae_AccumM GthetaAEAcc,
                            dae_AccumM GthetaSAEAcc,
                            DoubleAccumulator Jtheta,
                            DoubleAccumulator m,
                            DoubleAccumulator sparsityValue,
                            double[] pj,
                            double p,
                            double[] KLg,
                            double beta,
                            dae_AccumV pjAESparseAcc) throws IOException {

        dae_feedback_sparse AEfeedbacksparse = new dae_feedback_sparse();


        AEfeedbacksparse.getAverageActivation(event, thetasTL, pjAESparseAcc, m);
        for(int j=0; j<pjAESparseAcc.value().length; j++){pj[j] = pjAESparseAcc.value()[j]/m.value();} //averaged hidden unit activation
        for(int j=0; j<pj.length; j++){
            double sp = ( p * log(p/pj[j]) ) + ( (1-p)*log( (1-p)/(1-pj[j]) ) );//regularisation term
            sparsityValue.add(sp);
            KLg[j] = beta * ( -(p/pj[j]) + ( (1-p)/(1-pj[j]) ) );
        }
        AEfeedbacksparse.getAutoencodersSparse(event, thetasL, thetasTL, thetasS, thetasST, GthetaAEAcc, GthetaSAEAcc, Jtheta, KLg, beta);
    }


}

/*
AEmethodCG:sparsegc 2.3454567707530285 2.345353170345501 0.002 0.051800203763718855
gc 2.345510673514667 2.3452992723054957 0.002 0.10570060458570119
gc 2.3454934273623635 2.3453164810144904 0.002 0.08847317393656517
gc 2.3453393003651124 2.3454706635222733 0.002 -0.06568157858044543
gc 2.345422865233185 2.345387075998906 0.002 0.01789461713963547
gc 2.345441485654751 2.345368456140814 0.002 0.036514756968530904
gc 2.3454355321251708 2.345374405201953 0.002 0.03056346160890122
gc 2.34538228197467 2.3454276619770167 0.002 -0.022690001173275576
gc 2.345430870709642 2.345379070502633 0.002 0.0259001035045614
gc 2.345457821501964 2.3453521208907198 0.002 0.05285030562207638
gc 2.3454492031048053 2.3453607299270502 0.002 0.04423658887753312
gc 2.3453721326636976 2.3454378142457917 0.002 -0.03284079104703963
gc 2.345433748494362 2.3453761927090184 0.002 0.028777892671749683
gc 2.345463693891918 2.3453462487687577 0.002 0.058722561580015764
gc 2.345454117317381 2.3453558137867168 0.002 0.049151765332045017
gc 2.3453684843509297 2.3454414638860506 0.002 -0.03648976756043254
[0.051800203763718855, 0.10570060458570119, 0.08847317393656517, -0.06568157858044543, 0.01789461713963547, 0.036514756968530904, 0.03056346160890122, -0.022690001173275576, 0.0259001035045614, 0.05285030562207638, 0.04423658887753312, -0.03284079104703963, 0.028777892671749683, 0.058722561580015764, 0.049151765332045017, -0.03648976756043254]
 gradientback:0.43288 gradientcheck:0.43288 difference:-0.00000
 */
