# Motion Detection

This project is a mobile application developed with React Native, integrating a deep learning model to detect user movements, such as walking or lying down. It aligns with my academic program.

### 📌 Features

- Real-time motion detection using smartphone sensors (accelerometer and gyroscope).

- Deep learning model based on **LSTM** for accurate sequence analysis.

- Simple and intuitive interface displaying real-time predictions.

- Deployment via **Expo (EAS)** for simplified installation.

### 🛠 Technologies Used

- **React Native** for mobile app development.

- **TensorFlow.js** for deep learning model integration on mobile.

- **Google Colab** for model training.

- **scikit-learn** for data preprocessing.

### 📌 Prerequisites

Before installing and running the project, make sure you have the following tools installed:

- **Node.js** - [Download & Install](https://nodejs.org/)
- **Yarn** - Install it globally using:
```bash
npm install -g yarn
```

### 🚀 Installation & Execution

1. Clone the repository
```bash
git clone https://github.com/damiencren/motion_prediction
cd motion_prediction
```

2. Install dependencies

```bash
yarn install
```

3. Start the application in development mode

```bash
npx expo start
```

### How to use it

To use your own models, place them in the assets/models/* folder and ensure they are referenced in *index.tsx*. 
> **💡 Note :** This example uses two files for the weights because they are too large to be in one part

```typescript
  const modelJson = require("../../assets/model/lstm/model.json");
  const modelWeights = [require("../../assets/model/lstm/group1-shard1of2.bin"),require("../../assets/model/lstm/group1-shard2of2.bin"),];
```

To work with the application in development mode, use the following command

> **💡 Note :** We use npx to avoid installing the Expo and EAS CLI.

```bash
npx expo start
```
If you want to generate the APK, you need to log into your EAS account using the command
```bash
npx eas login
```
Then, create the project with this command
```bash
npx eas init
```
And use this command to build the APK
```bash
npx eas build --profile development --platform android 
```
The APK should be accessible on your EAS portal at https://expo.dev/. Use the APK to install the application on your phone and then open it.

<p align="center">
  <img src="docs/screen1.jpg" width="30%" />
  <img src="docs/screen2.jpg" width="30%" />
  <img src="docs/screen3.jpg" width="30%" />
</p>

### 📌 Future Improvements

- Optimizing model performance on mobile.

- Exploring hybrid architectures combining CNN + LSTM.

- Improved calibration for better prediction accuracy.

### 📜 License

- This project is licensed under the MIT License.
