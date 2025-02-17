import "@tensorflow/tfjs-react-native";
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";
import React, { useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Accelerometer, Gyroscope, AccelerometerMeasurement } from 'expo-sensors';
import ButterworthFilter from '../filters/butterworthFilter';
import * as tf from '@tensorflow/tfjs';


const SensorExample = () => {

  // Load the lstm model
  const modelJson = require("../../assets/model/lstm/model.json");
  const modelWeights = [require("../../assets/model/lstm/group1-shard1of2.bin"),require("../../assets/model/lstm/group1-shard2of2.bin"),];

  // Load the cnn model
  // const modelJson = require("../../assets/model/cnn/model.json");
  // const modelWeights = require("../../assets/model/cnn/group1-shard1of1.bin");

  const BUFFER_SIZE = 128; // time_steps

  // buffer for sensor data
  const dataBuffer = useRef<number[][]>([]);
  const bufferLock = useRef(false);

  const [accelerometerData, setAccelerometerData] = useState({ x: 0, y: 0, z: 0 });
  const [gyroscopeData, setGyroscopeData] = useState({ x: 0, y: 0, z: 0 });
  const [bodyAccelerometerData, setBodyAccelerometerData] = useState({ x: 0, y: 0, z: 0 });
  const [predictedGesture, setPredictedGesture] = useState<number | null>(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  // mean and std from the training
  const mean = [0.804749279, 0.0287554865, 0.0864980163, -0.000636303058, -0.000292296856, -0.000275299412, 0.000506464674, -0.000823780831, 0.000112948439];
  const std = [0.41411195, 0.39099543, 0.35776881, 0.19484634, 0.12242748, 0.10687881, 0.40681506, 0.38185432, 0.25574314];

  const gestures = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'];

  // resfesh rate for sensors
  Accelerometer.setUpdateInterval(20);
  Gyroscope.setUpdateInterval(20);

  const modelRef = useRef<tf.GraphModel | null>(null);

  const gravityFilter = useRef({
    x: new ButterworthFilter(0.3, 50),
    y: new ButterworthFilter(0.3, 50),
    z: new ButterworthFilter(0.3, 50)
  });

  // model loading
  useEffect(() => {
    const loadModel = async () => {
      await tf.ready();
      console.log('TensorFlow prêt');

      try {
        const model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
        modelRef.current = model;
        setIsModelLoaded(true);
        console.log('Modèle chargé avec succès');
      } catch (error) {
        console.error("Error loading model:", error);
      }
    };

    loadModel();
  }, []);

  useEffect(() => {
    const accelerometerSubscription = Accelerometer.addListener(handleAccelerometer);
    const gyroscopeSubscription = Gyroscope.addListener(setGyroscopeData);

    return () => {
      accelerometerSubscription.remove();
      gyroscopeSubscription.remove();
    };
  }, [gyroscopeData]);

  const handleAccelerometer = (data: AccelerometerMeasurement) => {
    const gravity = {
      x: gravityFilter.current.x.process(data.x),
      y: gravityFilter.current.y.process(data.y),
      z: gravityFilter.current.z.process(data.z)
    };

    const bodyAcc = {
      x: data.x - gravity.x,
      y: data.y - gravity.y,
      z: data.z - gravity.z
    };

    setBodyAccelerometerData(bodyAcc);

    const newData = [
      data.x, data.y, data.z,
      bodyAcc.x, bodyAcc.y, bodyAcc.z,
      gyroscopeData.x, gyroscopeData.y, gyroscopeData.z
    ];

    if (!bufferLock.current) {
      bufferLock.current = true;
      dataBuffer.current = [...dataBuffer.current, newData].slice(-BUFFER_SIZE);
      bufferLock.current = false;
    }

    setAccelerometerData(data);
  };

  // Prédiction périodique
  useEffect(() => {
    const predictInterval = setInterval(async () => {

      if (isModelLoaded && dataBuffer.current.length >= BUFFER_SIZE && modelRef.current) {
        bufferLock.current = true;
        try {
          const window = dataBuffer.current.slice(-BUFFER_SIZE);
          await predict(window);

          dataBuffer.current = dataBuffer.current.slice(-64);
        } catch (error) {
          console.error('Prediction error:', error);
        }
        bufferLock.current = false;
      }
    }, 1000); // 1sec

    return () => clearInterval(predictInterval);
  }, [isModelLoaded]);

  const predict = async (window: number[][]) => {
    try {
      const inputTensor = tf.tensor3d([window]);

      // Normalisation
      const processedTensor = inputTensor
        .sub(tf.tensor(mean))
        .div(tf.tensor(std));

      // Prédiction
      const prediction = modelRef.current!.predict(processedTensor);

      if (prediction instanceof tf.Tensor) {
        const output = await prediction.data();
        const predictedIndex = Array.from(output).indexOf(Math.max(...output));
        setPredictedGesture(predictedIndex);
        console.log('Predicted gesture:', gestures[predictedIndex]);
        tf.dispose(prediction);
      }
    } catch (error: any) {
      console.error('Erreur lors de la prédiction:', error.message);
      console.error('Détails:', JSON.stringify(error, Object.getOwnPropertyNames(error)));
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Sensor Data</Text>
      <Text style={styles.label}>Accelerometer Data</Text>
      <Text style={styles.data}>X: {accelerometerData.x}</Text>
      <Text style={styles.data}>Y: {accelerometerData.y}</Text>
      <Text style={styles.data}>Z: {accelerometerData.z}</Text>

      <Text style={styles.label}>Body Accelerometer Data</Text>
      <Text style={styles.data}>X: {bodyAccelerometerData.x}</Text>
      <Text style={styles.data}>Y: {bodyAccelerometerData.y}</Text>
      <Text style={styles.data}>Z: {bodyAccelerometerData.z}</Text>

      <Text style={styles.label}>Gyroscope Data</Text>
      <Text style={styles.data}>X: {gyroscopeData.x}</Text>
      <Text style={styles.data}>Y: {gyroscopeData.y}</Text>
      <Text style={styles.data}>Z: {gyroscopeData.z}</Text>

      {predictedGesture !== null ?(
        <View>
          <Text style={styles.label}>Predicted Gesture : {gestures[predictedGesture]}</Text>
        </View>
      ): (
        <View>
          
          <Text style={styles.label}>Loading Model...</Text>
        </View>
      )}

    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  label: {
    fontSize: 18,
    marginTop: 10,
  },
  data: {
    fontSize: 16,
  },
  tinyLogo: {
    width: 50,
    height: 50,
  },
});

export default SensorExample;
