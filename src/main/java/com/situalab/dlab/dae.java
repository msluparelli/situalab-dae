package com.situalab.dlab;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.util.DoubleAccumulator;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.lang.Math.log;


public class dae implements Serializable {


    public static void main(String[] args) throws Exception {


        //get Arguments
        String[] daeArgs = getArgs(args).split(";");
        Map<String,String> aeparams = getAEparams(daeArgs);
        for (int m=0; m<daeArgs.length; m++){
            String key = daeArgs[m].split(":")[0];
            String val = daeArgs[m].split(":")[1];
            aeparams.replace(key,val);
        }
        System.out.println("DAE params: "+aeparams);
        String datafile = aeparams.get("datafile").toString();
        String dnn = aeparams.get("dnn").toString(); int [] dnnA = Stream.of(dnn.split(",")).mapToInt(Integer::parseInt).toArray();
        String CG = aeparams.get("checkGradients").toString(); String AEmethodCG = aeparams.get("AEmethod").toString();

        //spark context
        SparkConf conf = new SparkConf().setAppName("deeplab");
        conf.setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        //read input data
        String dlab = datafile;
        String trainfilePath = "hdfs://localhost:9000/user/situalab/input/"+dlab+"train"; //train
        JavaRDD<LabeledPoint> traindata = MLUtils.loadLibSVMFile(jsc.sc(), trainfilePath).toJavaRDD();
        traindata.persist(StorageLevel.MEMORY_AND_DISK_SER());

        //classes
        dae_thetas AEthetas = new dae_thetas();
        dae_gradientcheck AEGradCheck = new dae_gradientcheck();

        //check gradients
        if(CG.equals("si")){AEGradCheck.checkGradient(traindata.first(), AEmethodCG, aeparams);}


        //thetas
        List<double[][]> thetasL = new ArrayList<double[][]>(); //thetas List


        int[] featN = {traindata.first().features().size()};
        int[] deepA = dnnA;
        int[] deepArch = IntStream.concat(Arrays.stream(featN), Arrays.stream(deepA)).toArray();
        for (int d=0; d<deepArch.length-1; d++){
            int row = deepArch[d];
            int col = deepArch[d+1];
            thetasL.add(AEthetas.getThetasAE(row,col));
            double[][] thetasAE = getthetasAEpretrained(jsc, traindata, thetasL, aeparams);
        }
        traindata.unpersist();
        jsc.close();


        //save thetas
        String nnthetaspath = "weightsAE/";
        //nnthetaspath="";
        saveThetas(thetasL, dlab, nnthetaspath);

    }

    private static double[][] getthetasAEpretrained(JavaSparkContext jsc,
                                                    JavaRDD<LabeledPoint> traindata,
                                                    List <double[][]> thetasL,
                                                    Map<String,String> aeparams){


        //methods
        dae_thetas AEthetas = new dae_thetas();
        //dae_feedback_regular_bug AEfeedback = new dae_feedback_regular_bug();
        dae_feedback_regular AEfeedbackregular = new dae_feedback_regular();
        dae_optimization AEoptimization = new dae_optimization();




        //Derivative architecture for AE back propagation, size of last thetas for back propagation
        int lasthetas = thetasL.size()-1; //last thetas to update
        int thetasrow = thetasL.get(lasthetas).length; //last thetas architecture
        int thetascol = thetasL.get(lasthetas)[0].length; //last thetas architecture
        double[][] thetas = new double[thetasrow][thetascol]; //thetas init
        double[][] Dthetas = new double[thetasrow][thetascol]; //derivative thetas, size of last thetas
        double[][] DthetasAcc = new double[thetasrow][thetascol]; //derivative thetas Acc for ADAlearning, size of last thetas

        //Derivative architecture for AE back propagation, size of thetas Star for back propagation
        double[][] thetasS = AEthetas.getthetasS(thetasL.get(lasthetas)); //only last thetas
        int thetasSrow = thetasS.length;
        int thetasScol = thetasS[0].length;
        double[][] DthetasS = new double[thetasSrow][thetasScol]; //derivative thetas, size of last thetas
        double[][] DthetasSAcc = new double[thetasSrow][thetasScol]; //derivative thetas Acc for ADAlearning, size of last thetas


        //params
        int epochS = Integer.parseInt(aeparams.get("epochs"));
        double learning = Double.parseDouble(aeparams.get("learning")); //learning rate
        double lmbda = Double.parseDouble(aeparams.get("lmbda")); //regularisation term
        String AEmethod = aeparams.get("AEmethod").toString();
        double p = Double.parseDouble(aeparams.get("sparsity")); //sparsity
        double beta = Double.parseDouble(aeparams.get("beta")); //sparsity regularisation penalty





        System.out.println("\ntraining:"+thetascol);
        for (int epoch = 0; epoch < epochS; epoch++) {

            //broadcas thetas & thetasT
            Broadcast<List<double[][]>> thetasLbd = jsc.broadcast(thetasL); //broadcast thetasL
            List<double[][]> thetasTL = AEthetas.getthetasTList(thetasL); //thetas transpose
            Broadcast<List<double[][]>> thetasTLbd = jsc.broadcast(thetasTL); //broadcast thetasTL
            Broadcast<double[][]> thetasSbd = jsc.broadcast(thetasS); //thetas Star
            double[][] thetasST = AEthetas.getTransposeAE(thetasS); //thetas Star Transpose
            Broadcast<double[][]> thetasSTbd = jsc.broadcast(thetasST);



            //dthetas Accumulator for backpropagation of last thetas architecture
            dae_AccumM GthetaAEAcc = new dae_AccumM(thetasrow, thetascol); //theta
            jsc.sc().register(GthetaAEAcc, "GradientthetaAEAcc"); //theta
            dae_AccumM GthetaSAEAcc = new dae_AccumM(thetasSrow, thetasScol); //thetaS
            jsc.sc().register(GthetaSAEAcc, "GradientthetaSAEAcc"); //thetaS
            DoubleAccumulator Jtheta = new DoubleAccumulator(); //J(theta) Accumulator
            jsc.sc().register(Jtheta, "Jtheta");
            DoubleAccumulator m = new DoubleAccumulator(); //J(theta) Accumulator
            jsc.sc().register(m, "mAE");
            DoubleAccumulator sparsityValue = new DoubleAccumulator(); //sparsity value Accumulator
            jsc.sc().register(sparsityValue, "sparsityvalue");
            dae_AccumV pjAESparseAcc = new dae_AccumV(thetascol); //include bias
            jsc.sc().register(pjAESparseAcc, "pj");

            //sparsity params
            double[] pj = new double[thetascol]; //include bias
            double[] KLg = new double[thetascol]; //include bias



            //auto encoder training, output -> GthetaAEAcc // Denoising Autoencoders
            double JthetaV = 0.0;
            if (AEmethod.equals("regular")|AEmethod.equals("denoising")){
                //traindata.foreach(xrdd -> AEfeedback.getAutoencoders(xrdd, thetasLbd.value(), thetasTLbd.value(), GthetaAEAcc, Jtheta, m, AEmethod));
                traindata.foreach(xrdd -> AEfeedbackregular.getAutoencoders(xrdd, thetasLbd.value(), thetasTLbd.value(), thetasSbd.value(), thetasSTbd.value(), GthetaAEAcc, GthetaSAEAcc, Jtheta, m, AEmethod));
                JthetaV = Jtheta.value()/m.value();
            }
            if (AEmethod.equals("sparse")){
                getAEsparse(traindata, thetasLbd, thetasTLbd, thetasSbd, thetasSTbd, GthetaAEAcc, GthetaSAEAcc, Jtheta, m, sparsityValue, pj, p, KLg, beta, pjAESparseAcc);
                JthetaV = Jtheta.value()/m.value() + beta*sparsityValue.value();
            }
            System.out.print("\repoch "+epoch+" J(theta) = "+JthetaV+" "+pjAESparseAcc.value().length+":hidden units:"+KLg.length);




            //optimization, Vanilla last thetas
            thetas = thetasL.get(thetasL.size()-1);
            Dthetas = AEoptimization.getderivativeAEBackProp(thetas, GthetaAEAcc, lmbda, m.value()); //derivative, gradients, verified
            DthetasAcc = AEoptimization.accumGradientsAE(Dthetas, DthetasAcc); //used for ADAGRAD
            thetas = AEoptimization.getthetasAEVanillaUpdated(thetas, Dthetas, learning, DthetasAcc, m.value(), lmbda); //vanillaUpdate, update
            thetasL = AEoptimization.updatethetasAEpretrained(thetasL, thetas); //update only last thetas

            //optimization, Vanilla thetas Star
            DthetasS = AEoptimization.getderivativeAEBackProp(thetasS, GthetaSAEAcc, lmbda, m.value()); //derivative, gradients, verified
            DthetasSAcc = AEoptimization.accumGradientsAE(DthetasS, DthetasSAcc); //used for ADAGRAD
            thetasS = AEoptimization.getthetasAEVanillaUpdated(thetasS, DthetasS, learning, DthetasSAcc, m.value(), lmbda); //vanillaUpdate, update


            //destroy broadcasted thetas
            thetasLbd.destroy(); //destroy broadcast at the end of loop
            thetasTLbd.destroy(); //destroy broadcast at the end of loop
            thetasSbd.destroy(); //destroy broadcast at the end of loop
            thetasSTbd.destroy(); //destroy broadcast at the end of loop
        }
        return thetas;
    }


    private static void getAEsparse(
            JavaRDD<LabeledPoint> traindata,
            Broadcast<List<double[][]>> thetasLbd,
            Broadcast<List<double[][]>> thetasTLbd,
            Broadcast<double[][]> thetasSbd,
            Broadcast<double[][]> thetasSTbd,
            dae_AccumM GthetaAEAcc,
            dae_AccumM GthetaSAEAcc,
            DoubleAccumulator Jtheta,
            DoubleAccumulator m,
            DoubleAccumulator sparsityValue,
            double[] pj,
            double p,
            double[] KLg,
            double beta,
            dae_AccumV pjAESparseAcc){

        dae_feedback_sparse AEfeedbacksparse = new dae_feedback_sparse();


        traindata.foreach(xrdd -> AEfeedbacksparse.getAverageActivation(xrdd, thetasTLbd.value(), pjAESparseAcc, m));
        for(int j=0; j<pjAESparseAcc.value().length; j++){pj[j] = pjAESparseAcc.value()[j]/m.value();} //averaged hidden unit activation
        for(int j=0; j<pj.length; j++){
            double sp = ( p * log(p/pj[j]) ) + ( (1-p)*log( (1-p)/(1-pj[j]) ) );//regularisation term
            sparsityValue.add(sp);
            KLg[j] = beta * ( -(p/pj[j]) + ( (1-p)/(1-pj[j]) ) );
        }
        traindata.foreach(xrdd -> AEfeedbacksparse.getAutoencodersSparse(xrdd, thetasLbd.value(), thetasTLbd.value(), thetasSbd.value(), thetasSTbd.value(), GthetaAEAcc, GthetaSAEAcc, Jtheta, KLg, beta));
    }




    //save AE trained thetas
    private static void saveThetas(List<double[][]> thetasL,
                                   String dlab,
                                   String nnthetaspath) throws IOException{

        //save thetas, architecture
        Charset charset = Charset.forName("US-ASCII");
        int[] autoEncoderArchitecture = new int[thetasL.size()+1];
        for (int layer=0; layer <thetasL.size(); layer++){
            autoEncoderArchitecture[layer] = thetasL.get(layer).length-1;
        }
        autoEncoderArchitecture[thetasL.size()] = thetasL.get(thetasL.size()-1)[0].length; //add output hidden unit layer
        int[] AEhiddenUnits = Arrays.copyOfRange(autoEncoderArchitecture, 1, thetasL.size()+1);
        String nnlayers = Arrays.toString(AEhiddenUnits).replaceAll("\\s+","");

        String nncompletethetaspath = nnthetaspath+dlab+nnlayers+"AE.csv";



        Path nnthetas = Paths.get(nncompletethetaspath); //path to save thetas
        BufferedWriter thetaswriter = Files.newBufferedWriter(nnthetas, charset); // thetas writter
        String thetasArchF ="";
        for (int a=0; a<autoEncoderArchitecture.length; a++) {
            thetasArchF += (Integer.toString(autoEncoderArchitecture[a])) + "_";
        }
        thetasArchF = thetasArchF.substring(0, thetasArchF.length()-1)+"_UnsupervisedAE\n";
        thetaswriter.write(thetasArchF); //write architecture

        dae_thetas AEthetas = new dae_thetas();
        double[] thetasflat = AEthetas.getflatthetasAE(thetasL);
        for (int t=0; t<thetasflat.length; t++){
            String theta = thetasflat[t]+"\n";
            thetaswriter.write(theta); //write thetas
        }
        thetaswriter.close(); //close file
        System.out.println();
        System.out.println(thetasArchF);
        System.out.println(nncompletethetaspath);

    }







    //DAE hyperparameters
    private static Map<String,String> getAEparams(String[] args){

        Map<String,String> daeparams = new HashMap<>();
        daeparams.put("datafile", ""); //input data
        daeparams.put("AEmethod", "denoising"); //denoising or sparse
        daeparams.put("epochs", "15"); //iterations
        daeparams.put("dnn", "4"); //dnn architecture
        daeparams.put("learning", "1.e-1"); //learning param
        daeparams.put("lmbda", "1.e-10"); //regularisation term
        daeparams.put("beta", "1.e-10"); //
        daeparams.put("sparsity", "0.05"); //denoising
        daeparams.put("RNN","no"); //RNN:yes
        daeparams.put("checkGradients", "no"); //check gradients
        return daeparams;
    }


    private static String getArgs(String[] args){

        String datos;

        try{
            datos = args[0];
        } catch (java.lang.ArrayIndexOutOfBoundsException e) {

            datos = "datafile:skitagGPStheOriginzs;dnn:24;checkGradients:si;AEmethod:denoising";
        }

        return datos;
    }



}
