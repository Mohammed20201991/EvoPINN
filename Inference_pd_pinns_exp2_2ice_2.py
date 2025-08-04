from google.colab import drive
drive.mount('/content/drive')

# test_model.py
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Custom layers Defintion (must match training exactly)
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.d1 = tf.keras.layers.Dense(units, activation='tanh')
        self.d2 = tf.keras.layers.Dense(units, activation='tanh')
        self.projection = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.projection = tf.keras.layers.Dense(self.units)
        super().build(input_shape)

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        return x + (self.projection(inputs) if self.projection else inputs)

class CollapsePINN(tf.keras.Model):
    def __init__(self, input_dim=6, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(128, activation='tanh')
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(64)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.output_layer(x)

# Load saved artifacts
def load_artifacts(model_path, scaler_X_path, scaler_y_path):
    model = load_model(model_path, custom_objects={
        'ResidualBlock': ResidualBlock,
        'CollapsePINN': CollapsePINN
    })
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_X, scaler_y

#  Prediction function
def predict(model, scaler_X, scaler_y, input_values):
    """Make prediction with preprocessing"""
    # Convert to numpy array
    input_array = np.array(input_values, dtype=np.float32).reshape(1, -1)

    # Scale input
    scaled_input = scaler_X.transform(input_array)

    # Predict
    scaled_pred = model.predict(scaled_input)

    # Inverse scale output
    return scaler_y.inverse_transform(scaled_pred)[0][0]

# Test with user input
if __name__ == "__main__":
    # Paths to saved files (update these)
    MODEL_PATH = "/content/drive/MyDrive/PINNs/PINNs_Exp2_2_ICE_2_MC/deployment_package/model.keras"
    SCALER_X_PATH = "/content/drive/MyDrive/PINNs/PINNs_Exp2_2_ICE_2_MC/deployment_package/scaler_X.joblib"
    SCALER_Y_PATH = "/content/drive/MyDrive/PINNs/PINNs_Exp2_2_ICE_2_MC/deployment_package/scaler_y.joblib"

    # Load model and scalers
    model, scaler_X, scaler_y = load_artifacts(MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH)

    # Get user input
    print("Enter prediction inputs:")
    suction = float(input("Suction (kPa): "))
    silica = float(input("Silica fume (%): "))
    lime = float(input("Lime (%): "))
    gypsum = float(input("Gypsum content (%): "))
    stress = float(input("Applied vertical stress (kPa): "))
    saturation = float(input("Degree of Saturation (%): "))

    # Make prediction
    pred = predict(model, scaler_X, scaler_y,
                  [suction, silica, lime, gypsum, stress, saturation])

    print(f"\nPredicted Collapse Potential: {pred:.2f}%")

# %% [markdown]
# #  Collapse Potential Predictor (Colab Version)
# Interactive demo for your PINN model

# %% [markdown]
# ### ⚙️ Setup

# %%
!pip install -q ipywidgets matplotlib
!jupyter nbextension enable --py widgetsnbextension

# %%
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Layout, VBox, HBox, Output
import IPython.display as display

# %% [markdown]
# ###  Load Model

# %%
# Define custom layers (must match training)
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.d1 = tf.keras.layers.Dense(units, activation='tanh')
        self.d2 = tf.keras.layers.Dense(units, activation='tanh')
        self.projection = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.projection = tf.keras.layers.Dense(self.units)
        super().build(input_shape)

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        return x + (self.projection(inputs) if self.projection else inputs)

class CollapsePINN(tf.keras.Model):
    def __init__(self, input_dim=6, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(128, activation='tanh')
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(64)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.output_layer(x)

# Load model and scalers
model = load_model(
    "/content/drive/MyDrive/PINNs/PINNs_Exp2_2_ICE_2_MC/deployment_package/model.keras",
    custom_objects={'ResidualBlock': ResidualBlock, 'CollapsePINN': CollapsePINN}
)
scaler_X = joblib.load("/content/drive/MyDrive/PINNs/PINNs_Exp2_2_ICE_2_MC/deployment_package/scaler_X.joblib")
scaler_y = joblib.load("/content/drive/MyDrive/PINNs/PINNs_Exp2_2_ICE_2_MC/deployment_package/scaler_y.joblib")

# %% [markdown]
# ### 🎮 Interactive Controls

# %%
# Prediction function
def predict(suction, silica, lime, gypsum, stress, saturation):
    inputs = np.array([[suction, silica, lime, gypsum, stress, saturation]])
    scaled_input = scaler_X.transform(inputs)
    scaled_pred = model.predict(scaled_input)
    return scaler_y.inverse_transform(scaled_pred)[0][0]

# Create output area for dynamic updates
out = Output()

# Slider styles
slider_layout = Layout(width='90%', height='50px')

# Create interactive widgets
suction_slider = FloatSlider(min=0, max=100, value=50, step=1, description="Suction (kPa):", layout=slider_layout)
silica_slider = FloatSlider(min=0, max=20, value=10, step=0.5, description="Silica (%):", layout=slider_layout)
lime_slider = FloatSlider(min=0, max=10, value=2, step=0.1, description="Lime (%):", layout=slider_layout)
gypsum_slider = FloatSlider(min=0, max=30, value=15, step=0.5, description="Gypsum (%):", layout=slider_layout)
stress_slider = FloatSlider(min=50, max=500, value=200, step=10, description="Stress (kPa):", layout=slider_layout)
saturation_slider = FloatSlider(min=30, max=100, value=80, step=1, description="Saturation (%):", layout=slider_layout)

# %% [markdown]
# ### 📊 Live Visualization

# %%
def update_plot(suction, silica, lime, gypsum, stress, saturation):
    with out:
        out.clear_output(wait=True)

        # Get prediction
        pred = predict(suction, silica, lime, gypsum, stress, saturation)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Input radar plot
        categories = ['Suction', 'Silica', 'Lime', 'Gypsum', 'Stress', 'Saturation']
        values = [suction/100, silica/20, lime/10, gypsum/30, stress/500, saturation/100]

        ax1.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F06292'])
        ax1.set_ylim(0, 1)
        ax1.set_title("Normalized Input Values")
        ax1.tick_params(axis='x', rotation=45)

        # Prediction display
        ax2.axis('off')
        ax2.text(0.5, 0.6, f"{pred:.2f}%",
                fontsize=40, ha='center', va='center', color='#1E88E5')
        ax2.text(0.5, 0.3, "Predicted Collapse Potential",
                fontsize=12, ha='center', va='center')

        plt.tight_layout()
        plt.show()

# Connect widgets to update function
interact(update_plot,
         suction=suction_slider,
         silica=silica_slider,
         lime=lime_slider,
         gypsum=gypsum_slider,
         stress=stress_slider,
         saturation=saturation_slider)

# Display the output area
display.display(out)

# %% [markdown]
# ###  How to Use:
# 1. Adjust the sliders above
# 2. Watch the visualization update in real-time
# 3. The right panel shows predicted collapse potential