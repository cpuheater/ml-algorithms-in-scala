package com.cpuheater.ml

import com.cpuheater.ml.supervised.LinearRegression
import com.cpuheater.ml.util.TestSupport
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

class LinearRegressionSpec  extends TestSupport{


  it should "test linear regression" in {

    val numLinesToSkip = 1
    val delimiter = ','
    val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReader.initialize(new FileSplit(new ClassPathResource("internet-users.csv").getFile))
    val iter:DataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000000,0,0, true)
    val dataSet: DataSet = iter.next()
    val x = dataSet.getFeatures()
    val y = dataSet.getLabels()

    val lr = 0.02f
    val iterations = 500

    val model = new LinearRegression()
    val params = model.fit(x, y, lr, iterations)
    println(s"Parameters theta_0 = ${params.getFloat(0)} theta_1 = ${params.getFloat(1)}")

    val year = 27f
    val pred = model.predict(Nd4j.create(Array(year)))
    println(s"The predicted number of internet users (per 100 people) in 20${year} is ${pred}")

  }


}
