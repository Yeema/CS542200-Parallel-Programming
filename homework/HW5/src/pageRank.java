package pageRank;
import java.io.IOException;
import java.util.StringTokenizer;
// hadoop jar WordCount.jar wordcount.WordCount in.txt out.txt
// .000000000000000020920000000000
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Partitioner;  
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.io.WritableComparable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;
import java.text.NumberFormat;
import java.text.DecimalFormat;
import java.net.URI; 
import java.io.*;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.io.DoubleWritable.Comparator;

public class pageRank {
    public static double alpha = 0.85;
    public static int N;
    public static int dangling_N;
    public static double danglingPR;
    public static double error;
    public static NumberFormat NF = new DecimalFormat("00");
    // output 
    // <to> <from> 
    // <to> <@@@@@@@@@@@@@@@>
    public static class parseRawMapper extends Mapper<LongWritable, Text, Text, Text> {
        public static enum N_Counter {
            N
        }
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            /*  Match title pattern */  
            Pattern titlePattern = Pattern.compile("<title>(.+?)</title>");
            String lines = value.toString();
            Matcher titleMatcher = titlePattern.matcher(lines);
            // No need capitalizeFirstLetter
            titleMatcher.find();
            Text title = new Text(unescapeXML(titleMatcher.group(1)));

            context.write(title, new Text("@@@@@@@@@@@@@@@"));
            context.getCounter(N_Counter.N).increment(1);
            /*  Match link pattern */
            Pattern linkPattern = Pattern.compile("\\[\\[(.+?)([\\|#]|\\]\\])");
            Matcher linkMatcher = linkPattern.matcher(lines);
            // Need capitalizeFirstLetter

            while(linkMatcher.find()){
                String s = capitalizeFirstLetter(unescapeXML(linkMatcher.group(1)));
                if(s.isEmpty()) continue;
                context.write(new Text(s), title);
            }
            
		}

        private String unescapeXML(String input) {

            return input.replaceAll("&lt;", "#").replaceAll("&gt;", ">").replaceAll("&amp;", "&").replaceAll("&quot;", "\"").replaceAll("&apos;", "\'");

        }

        private String capitalizeFirstLetter(String input){
            char firstChar = input.charAt(0);

            if ( firstChar >= 'a' && firstChar <='z'){
                if ( input.length() == 1 ){
                    return input.toUpperCase();
                }
                else
                    return input.substring(0, 1).toUpperCase() + input.substring(1);
            }
            else 
                return input;
        }
	}
    public static class parseRawCombiner extends Reducer<Text, Text, Text, Text> {
    
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            StringBuffer links = new StringBuffer();
            boolean first = true;
			for(Text value:values){
                String val = value.toString();
                if(val.contains("@@@@@@@@@@@@@@@")){
                    context.write(key,value);
                }else{
                    if(!first)
                        links.append("<");
                    links.append(val);
                    first = false;
                }
            }
            if(!first)
                context.write(key,new Text(links.toString()));
		}

	}
    public static class parseRawPartitioner extends Partitioner<Text,Text> {
        @Override
        public int getPartition(Text key, Text value, int numReduceTasks) {
            return Math.abs(key.hashCode()% numReduceTasks);
        }
    }
    // output
    // <to> <node1><<node2><<node3>
    // <to>
	public static class parseRawReducer extends Reducer<Text, Text, Text, Text> {
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			boolean first = true;
			StringBuffer links = new StringBuffer();
            boolean valid = false;

            for(Text value: values){
                String val = value.toString();
                if(val.equals("@@@@@@@@@@@@@@@")){
                    valid = true;
                }else{
                    if(!first){
                        links.append("<");
                    }
                    links.append(val);
                    first = false;
                }
            }
            if(valid)
                context.write(key,new Text(links.toString()));
		}
	}
    // input
    // <to> <node1><<node2><<node3>
    // <to>
    // output
    // only when tos exist <from> <to1><<to2><<to3>
    // every node had      <from> <@@@@@@@@@@@@@@@>
    public static class reverseMapper extends Mapper<Text, Text, Text, Text> {
		@Override
		public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            String lines = value.toString();

            context.write(key, new Text("@@@@@@@@@@@@@@@"));
            if(!lines.isEmpty()){
                String[] links  = lines.split("<");
                for(String link : links){
                    if(!link.isEmpty()){
                        context.write(new Text(link),key);
                    }
                }
            }
		}   
	}
    public static class reverseCombiner extends Reducer<Text, Text, Text, Text> {
    
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            StringBuffer links = new StringBuffer();
            boolean first = true;
			for(Text value:values){
                String val = value.toString();
                if(val.equals("@@@@@@@@@@@@@@@")){
                    context.write(key,value);
                }else{
                    if(!first)
                        links.append("<");
                    links.append(val);
                    first = false;
                }
            }
            if(!first)
                context.write(key,new Text(links.toString()));
		}

	}
    public static class reversePartitioner extends Partitioner<Text,Text> {
        @Override
        public int getPartition(Text key, Text value, int numReduceTasks) {
            return Math.abs(key.hashCode()% numReduceTasks);
        }
    }
    // input
    // only when tos exist <from> <to1><<to2><<to3>
    // every node had      <from> <@@@@@@@@@@@@@@@>
    //output
    // with outlinks    <from> <pr><<to1><<to2>
    // without outlinks <from> <pr>
    public static class reverseReducer extends Reducer<Text, Text, Text, Text> {
        public static enum Dangling_Counter {
            dangling_N
        }
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            boolean first = true;
            StringBuffer links = new StringBuffer();
            int N = context.getConfiguration().getInt("N", 0);
            double rank = (double)1.0/(double)N;
            links.append(String.valueOf(rank));
            for(Text value: values){
                String val = value.toString();
                if(!val.equals("@@@@@@@@@@@@@@@")){
                    links.append("<");
                    links.append(val);
                    first = false;
                }
            }
            context.write(key,new Text(links.toString()));
            if(first){
                context.getCounter(Dangling_Counter.dangling_N).increment(1);
            }
		}
	}
    //input
    // with outlinks    <from> <pr><<to1><<to2>
    // without outlinks <from> <pr>
    // output
    // link info: <from> <pr><<to1><<to2> or <from> <pr>
    // pagerank: <to> @@<pr/<to's outlink>
    public static class countRankMapper extends Mapper<Text, Text, Text, Text> {
		@Override
		public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            String lines = value.toString();
            if(lines.contains("<")){
                String[] links = lines.split("<");
                double rank = (double)links.length - 1.0;
                for(int i =1 ; i<links.length ; i++){
                    context.write(new Text(links[i]),new Text("$#"+String.valueOf(Double.parseDouble(links[0])/rank)));
                }
            }
            context.write(key,value);
		}   
	}
    // input
    // link info: <from> <pr><<to1><<to2> or <from> <pr>
    // pagerank: <to> @@<pr/<to's outlink>
    public static class countRankCombiner extends Reducer<Text, Text, Text, Text> {
    
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			double links = 0.0;

			for (Text value : values) {
				String val = value.toString();
                if(!val.substring(0,2).equals("$#") )
                    context.write(key,value);
                else{
                    links += Double.parseDouble(val.substring(2));
                }
			}
            if(links>0.0)
			    context.write(key, new Text("$#"+String.valueOf(links)));
        }
	}
    public static class countRankPartitioner extends Partitioner<Text,Text> {
        @Override
        public int getPartition(Text key, Text value, int numReduceTasks) {
            return Math.abs(key.hashCode()% numReduceTasks);
        }
    }
    // input
    // link info: <from> <pr><<to1><<to2> or <from> <pr>
    // pagerank: <to> @<pr/<to's outlink>
    // output
    // with outlinks    <from> <pr><<to1><<to2>
    // without outlinks <from> <pr>
    public static class countRankReducer extends Reducer<Text, Text, Text, Text> {
        public static enum Hyper_Counter{
            error,danglingPR,danglingN,NN
        }
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String pages=null;
            double ranks=0.0;
            double links = 0.0;
            double summation = 0.0;
			for (Text value : values) {
                String val = value.toString();
                //with link info
                if(!val.substring(0,2).equals("$#") ){
                    if(val.contains("<")){
                        int tIdx1 = val.indexOf("<");
                        ranks = Double.parseDouble(val.substring(0,tIdx1));
                        pages = val.substring(tIdx1);
                    }else{
                        ranks = Double.parseDouble(val);
                    }
                }else{// only rank
                    links += Double.parseDouble(val.substring(2));
                }
			}
            Double danglingNodePR = Double.parseDouble(context.getConfiguration().get("danglingPR"));
            int N = (int)context.getConfiguration().getInt("N", 0);
            summation = (1.0-alpha)/(double)N+ alpha*links + alpha*danglingNodePR/(double)N;
            
            if(pages!=null)
                context.write(key,new Text(String.valueOf(summation)+pages));
            else{
                context.write(key,new Text(String.valueOf(summation)));
                context.getCounter(Hyper_Counter.danglingPR).increment((long) (summation*Math.pow(2,63)));
                context.getCounter(Hyper_Counter.danglingN).increment(1);
            }   
            context.getCounter(Hyper_Counter.error).increment((long)(Math.abs(summation-ranks)*Math.pow(2,63)));
            context.getCounter(Hyper_Counter.NN).increment(1);
		}

	}
    public static class SortPair implements WritableComparable{
        private Text word;
        private double PageRank;
        public int N;
        public SortPair() {
            word = new Text();
            PageRank = 0.0;
            N = 0;
        }

        public SortPair(Text word, double PageRank,int N) {
            //TODO: constructor
            this.word = word;
            this.PageRank = PageRank;
            this.N = N;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            this.word.write(out);
            out.writeDouble(this.PageRank);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            this.word.readFields(in);
            this.PageRank = in.readDouble();
        }

        public Text getWord() {
            return this.word;
        }

        public double getPageRank() {
            return this.PageRank;
        }

        @Override
        public int compareTo(Object o) {

            double thisPageRank = this.getPageRank();
            double thatPageRank = ((SortPair)o).getPageRank();

            Text thisWord = this.getWord();
            Text thatWord = ((SortPair)o).getWord();

            // Compare between two objects
            // First order by PageRank, and then sort them lexicographically in ascending order
            if(Double.compare(thisPageRank, thatPageRank)!=0){
                if(Double.compare(thisPageRank, thatPageRank)>0)
                    return -1;
                else
                    return 1;
            }
            return thisWord.toString().compareTo(thatWord.toString());
        }
    } 
    // input
    // with outlinks    <from> <pr><<to1><<to2>
    // without outlinks <from> <pr>
    // output
    // <sortpair> <null>
    public static class sortRankMapper extends Mapper<Text, Text, SortPair, NullWritable> {
        public static enum N_Counter {
            N
        }
		@Override
		public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            int N = (int)context.getConfiguration().getInt("N", 0);
            String lines = value.toString();
            if(lines.contains("<")){
                String[] links = lines.split("<");
                SortPair pair = new SortPair(key,Double.parseDouble(links[0]),N);
                context.write(pair,NullWritable.get());
            }else{
                SortPair pair = new SortPair(key,Double.parseDouble(lines),N);
                context.write(pair,NullWritable.get());
            }
            context.getCounter(N_Counter.N).setValue(N);
		}   
	}

    public static class sortRankPartitioner extends Partitioner<SortPair, NullWritable> {    
        @Override
        public int getPartition(SortPair key, NullWritable value, int numReduceTasks) {
            Double interval = 1.0/(double)key.N;
            int level = (numReduceTasks/2-1) - (int)(key.getPageRank()/interval);
            if(level<0) level = 0;
            if(level == numReduceTasks/2 -1){
                interval = interval/20.0;
                int level_plum = (numReduceTasks/2-1) - (int)((key.getPageRank())/interval);
                if(level_plum<0) level_plum = 0;
                level += level_plum;
            }
            return level;
        }
    }
    public static class sortRankReducer extends Reducer<SortPair, NullWritable, Text, DoubleWritable> {
        public void reduce(SortPair key, Iterable<NullWritable> values, Context context)throws java.io.IOException, InterruptedException {
            
            context.write(key.getWord(),new DoubleWritable(key.getPageRank()));
        }
    }

    public static void parseRaw(String in, String out) throws IOException, ClassNotFoundException, InterruptedException {
        
        Job job = Job.getInstance(new Configuration(), "parseRaw");
        job.setJarByClass(pageRank.class);
        
        // input / mapper
        FileInputFormat.addInputPath(job, new Path(in));
        job.setInputFormatClass(TextInputFormat.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setMapperClass(parseRawMapper.class);
        job.setCombinerClass(parseRawCombiner.class);
        job.setPartitionerClass(parseRawPartitioner.class);
        // output / reducer
        FileOutputFormat.setOutputPath(job, new Path(out));
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setReducerClass(parseRawReducer.class);
        job.setNumReduceTasks(32);
        job.waitForCompletion(true);
        pageRank.N = (int) job.getCounters()
				.findCounter(parseRawMapper.N_Counter.N)
				.getValue();
    }
    
    public static void reverse(String in, String out) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        conf.setInt("N", pageRank.N);
        Job job = Job.getInstance(conf, "reverse");
        job.setJarByClass(pageRank.class);
        
        // input / mapper
        FileInputFormat.addInputPath(job, new Path(in));
        job.setInputFormatClass(KeyValueTextInputFormat.class);	
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setMapperClass(reverseMapper.class);
        job.setCombinerClass(reverseCombiner.class);
        job.setPartitionerClass(reversePartitioner.class);
        // output / reducer
        FileOutputFormat.setOutputPath(job, new Path(out));
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setReducerClass(reverseReducer.class);
        job.setNumReduceTasks(32);
        job.waitForCompletion(true);
        pageRank.dangling_N = (int)job.getCounters()
				.findCounter(reverseReducer.Dangling_Counter.dangling_N)
				.getValue();
     
    }
    public static void countRank(String in, String out) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
		conf.set("danglingPR", String.valueOf(pageRank.danglingPR));
		conf.setInt("N", pageRank.N);
        Job job = Job.getInstance(conf, "countRank");
        job.setJarByClass(pageRank.class);
        
        // input / mapper
        FileInputFormat.addInputPath(job, new Path(in));
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setMapperClass(countRankMapper.class);
        job.setCombinerClass(countRankCombiner.class);
        // output / reducer
        FileOutputFormat.setOutputPath(job, new Path(out));
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setPartitionerClass(countRankPartitioner.class);
        job.setReducerClass(countRankReducer.class);
        job.setNumReduceTasks(32);
        job.waitForCompletion(true);
        pageRank.error = (double)((long)job.getCounters()
				.findCounter(countRankReducer.Hyper_Counter.error)
				.getValue()/Math.pow(2,63));
        pageRank.danglingPR = (double)((long)job.getCounters()
				.findCounter(countRankReducer.Hyper_Counter.danglingPR)
				.getValue()/Math.pow(2,63));
    }
    public static void sortRank(String in, String out)throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        conf.setInt("N", pageRank.N);

        Job job = Job.getInstance(conf, "sortRank");
        job.setJarByClass(pageRank.class);
        
        // input / mapper
        FileInputFormat.addInputPath(job, new Path(in));
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        job.setMapOutputKeyClass(SortPair.class);
        job.setMapOutputValueClass(NullWritable.class);
        job.setMapperClass(sortRankMapper.class);
        // output / reducer
        FileOutputFormat.setOutputPath(job, new Path(out));
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        job.setPartitionerClass(sortRankPartitioner.class);
        job.setReducerClass(sortRankReducer.class);
        job.setNumReduceTasks(32);
        job.waitForCompletion(true);
     
    }
    
    public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
        String experiment = "";
        int iteration = 0;
		int iterationLimit = -1;
		pageRank pagerank = new pageRank();
        //set arguments
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        String input = otherArgs[0];
        String outpath = otherArgs[1];
        if (otherArgs.length == 3){
            iterationLimit = Integer.parseInt(otherArgs[2]);
        }
        //iter00 parseRaw->reverse->count pagerank
		pagerank.parseRaw(otherArgs[0],outpath+"/iter00/parse");
        pagerank.reverse(outpath+"/iter00/parse/p*",outpath+"/iter00/reverse");
        Path path;
        path = new Path(outpath + "/iter00/parse");
        if(fs.exists(path)){
            fs.delete(path,true);
        }

        
        // System.out.println("num of N: "+N+"\n"+"num of dangling: "+dangling_N);
        
        danglingPR = (double)dangling_N/(double)N;
        pageRank.countRank(outpath+"/iter00/reverse/p*",outpath+"/iter01/input");
        
        path = new Path(outpath + "/iter00/reverse");
        if(fs.exists(path)){
            fs.delete(path,true);
        }
        // System.out.println("error: "+error);
        experiment += "(iteration , error): ("+iteration +" , "+error+")\n";
        if(iterationLimit==-1){
            iteration = 1;
            while(error>0.001){
                input = outpath + "/iter" + NF.format(iteration)+"/input/p*";
                String output = outpath + "/iter" + NF.format(iteration+1);
                // count dangling, count page rank
                pageRank.countRank(input,output+"/input");
                path = new Path(outpath + "/iter"+NF.format(iteration));
                if(fs.exists(path)){
                    fs.delete(path,true);
                }
                iteration+=1;
                // System.out.println("error: "+error);
                experiment += "(iteration , error): ("+iteration+" , "+error+")\n";
            }
        }else{
            for(iteration=1 ; iteration<iterationLimit ; iteration++){
                input = outpath + "/iter" + NF.format(iteration)+"/input/p*";
                String output = outpath + "/iter" + NF.format(iteration+1);

                pageRank.countRank(input,output+"/input");
                path = new Path(outpath + "/iter"+NF.format(iteration));
                if(fs.exists(path)){
                    fs.delete(path,true);
                }
                // System.out.println("iteration: "+iteration);
                // System.out.println("error: "+error);
                experiment += "(iteration , error): ("+iteration+" , "+error+")\n";
            }
        }
        pageRank.sortRank(outpath + "/iter" + NF.format(iteration)+"/input/p*",outpath+"/result");
        // System.out.println("iteration: "+iteration+"\nerror: "+error);
        // experiment += "(iteration , error): ("+String.valueOf(iteration)+" , "+String.valueOf(error)+")";
        // System.out.println(experiment);
	}    
}