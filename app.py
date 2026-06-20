

import tensorflow as tf

import cv2
import numpy as np
import gradio as gr
from PIL import Image
from tensorflow.keras.models import load_model

IMG_SIZE = 224

# Load trained model (must be uploaded to HF)
model = load_model("skin_model.h5")

# IMPORTANT: Must match training class order EXACTLY
CLASSES = ['acne', 'dry', 'oily', 'hyperpigmentation', 'normal']



# Preprocessing function
def preprocess_image(img_input):
    try:
        if isinstance(img_input, np.ndarray):
            img_rgb = img_input
        else:
            img = cv2.imread(img_input)
            if img is None:
                return None, "Error loading image"
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        face_detected = False
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            img_rgb = img_rgb[y:y+h, x:x+w]
            face_detected = True

        img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_rgb = np.expand_dims(img_rgb / 255.0, axis=0)
        return img_rgb, "Face detected" if face_detected else "No face detected - analyzing full image"

    except Exception as e:
        return None, f"Error: {str(e)}"

# FIXED: Complete recommendations database
def get_recommendations(condition):
    """
    Returns detailed recommendations - GUARANTEED to return something
    """

    all_recommendations = {
        'acne': {
            'emoji': '🔴',
            'title': 'Acne-Prone Skin',
            'morning': [
                '**Cleanser:** Salicylic Acid 2% foaming cleanser',
                '**Toner:** Niacinamide or witch hazel toner',
                '**Treatment:** Benzoyl peroxide 2.5% spot treatment',
                '**Moisturizer:** Oil-free gel moisturizer',
                '**Sunscreen:** Mineral SPF 50 (zinc oxide)'
            ],
            'night': [
                '**Cleanser:** Same as morning (double cleanse if makeup)',
                '**Exfoliant:** Salicylic acid 2% (3x/week)',
                '**Treatment:** Adapalene or retinol (start 2x/week)',
                '**Moisturizer:** Light, non-comedogenic moisturizer'
            ],
            'products': [
                'CeraVe Foaming Facial Cleanser',
                'Paula\'s Choice 2% BHA Liquid Exfoliant',
                'The Ordinary Niacinamide 10% + Zinc 1%',
                'Differin Gel (Adapalene 0.1%)',
                'Neutrogena Hydro Boost Water Gel'
            ],
            'tips': [
                '💧 Drink 8+ glasses water daily',
                '🛏️ Change pillowcases 2-3x per week',
                '🚫 Avoid touching/picking at face',
                '🥗 Reduce dairy and high-glycemic foods'
            ]
        },
        'dry': {
            'emoji': '💧',
            'title': 'Dry Skin',
            'morning': [
                '**Cleanser:** Cream-based gentle cleanser',
                '**Toner:** Hydrating toner with hyaluronic acid',
                '**Serum:** Hyaluronic acid serum (on damp skin)',
                '**Moisturizer:** Rich cream with ceramides',
                '**Sunscreen:** Moisturizing SPF 50'
            ],
            'night': [
                '**Cleanser:** Creamy cleanser or micellar water',
                '**Serum:** Vitamin E or squalane oil',
                '**Moisturizer:** Thick night cream or sleeping mask',
                '**Face Oil:** Rosehip or argan oil (optional)'
            ],
            'products': [
                'CeraVe Hydrating Cleanser',
                'The Ordinary Hyaluronic Acid 2% + B5',
                'La Roche-Posay Toleriane Double Repair',
                'Aquaphor Healing Ointment (for dry patches)',
                'Laneige Water Sleeping Mask'
            ],
            'tips': [
                '💧 Drink 10+ glasses water daily',
                '🏠 Use a humidifier (especially winter)',
                '🚿 Use lukewarm water (not hot)',
                '🥑 Eat omega-3 rich foods'
            ]
        },
        'oily': {
            'emoji': '✨',
            'title': 'Oily Skin',
            'morning': [
                '**Cleanser:** Gel-based foaming cleanser',
                '**Toner:** Witch hazel or niacinamide toner',
                '**Serum:** Niacinamide 10% for oil control',
                '**Moisturizer:** Lightweight gel moisturizer',
                '**Sunscreen:** Matte-finish, oil-free SPF 50'
            ],
            'night': [
                '**Cleanser:** Same as morning',
                '**Exfoliant:** BHA/AHA (2-3x per week)',
                '**Serum:** Retinol or niacinamide',
                '**Moisturizer:** Light gel (oily skin needs moisture!)',
                '**Weekly:** Clay mask on T-zone'
            ],
            'products': [
                'CeraVe Foaming Facial Cleanser',
                'The Ordinary Niacinamide 10% + Zinc 1%',
                'Paula\'s Choice 2% BHA Liquid',
                'Neutrogena Hydro Boost Water Gel',
                'Aztec Secret Indian Healing Clay'
            ],
            'tips': [
                '💧 Stay hydrated (helps balance oil)',
                '📄 Use blotting papers (don\'t over-wash)',
                '🥗 Reduce fried/greasy foods',
                '🧴 Wash makeup brushes weekly'
            ]
        },
        'hyperpigmentation': {
            'emoji': '🎨',
            'title': 'Hyperpigmentation',
            'morning': [
                '**Cleanser:** Brightening cleanser with vitamin C',
                '**Serum:** Vitamin C serum 15-20%',
                '**Treatment:** Alpha arbutin or tranexamic acid',
                '**Moisturizer:** Light moisturizer with niacinamide',
                '**Sunscreen:** SPF 50+ CRITICAL! Reapply every 2 hours'
            ],
            'night': [
                '**Cleanser:** Same as morning',
                '**Exfoliant:** AHA (glycolic/lactic acid) 3-4x/week',
                '**Serum:** Retinol 0.25-1%',
                '**Treatment:** Niacinamide or alpha arbutin',
                '**Moisturizer:** Hydrating cream'
            ],
            'products': [
                'CeraVe Vitamin C Serum',
                'The Ordinary Alpha Arbutin 2% + HA',
                'Paula\'s Choice 10% Azelaic Acid Booster',
                'Differin Gel (Adapalene)',
                'EltaMD UV Clear SPF 46 (tinted)'
            ],
            'tips': [
                '☀️ SUNSCREEN DAILY (non-negotiable!)',
                '🧢 Wear hats, seek shade 10am-4pm',
                '⏰ Be patient - fading takes 6-12 months',
                '🚫 Don\'t pick at skin (causes more PIH)'
            ]
        },
        'normal': {
            'emoji': '✅',
            'title': 'Normal/Balanced Skin',
            'morning': [
                '**Cleanser:** Gentle gel or cream cleanser',
                '**Serum:** Vitamin C for antioxidant protection',
                '**Moisturizer:** Balanced moisturizer',
                '**Sunscreen:** SPF 30-50 broad spectrum'
            ],
            'night': [
                '**Cleanser:** Same as morning',
                '**Exfoliant:** 2-3x per week (AHA or BHA)',
                '**Serum:** Retinol or peptides (anti-aging)',
                '**Moisturizer:** Night cream'
            ],
            'products': [
                'CeraVe Hydrating Facial Cleanser',
                'The Ordinary Vitamin C 23% + HA',
                'Cetaphil Daily Facial Moisturizer SPF 50',
                'The Ordinary Retinol 0.5% in Squalane',
                'Neutrogena Hydro Boost Gel Cream'
            ],
            'tips': [
                '💧 Maintain hydration (8+ glasses)',
                '😴 Get 7-9 hours quality sleep',
                '🥗 Balanced diet with antioxidants',
                '🏃 Regular exercise for circulation'
            ]
        }
    }

    # Return recommendation or fallback to normal
    return all_recommendations.get(condition.lower(), all_recommendations['normal'])

# Enhanced prediction function
def predict_skin(image):
    if image is None:
        return """
        <div style='text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px;'>
            <h2>⚠️ No Image Captured</h2>
            <p style='font-size: 18px;'>Please use the webcam to capture your face</p>
        </div>
        """

    # Process image
    processed, status = preprocess_image(image)

    if processed is None:
        return f"""
        <div style='text-align: center; padding: 50px; background: #ff4444; color: white; border-radius: 15px;'>
            <h2>❌ Error Processing Image</h2>
            <p style='font-size: 18px;'>{status}</p>
            <p>Please try again with better lighting</p>
        </div>
        """

    # Make prediction
    pred = model.predict(processed, verbose=0)[0]
    class_idx = np.argmax(pred)
    confidence = pred[class_idx]
    condition = CLASSES[class_idx]

    # Get recommendations
    rec = get_recommendations(condition)

    # Generate confidence bars for all classes
    confidence_bars = ""
    for i, class_name in enumerate(CLASSES):
        percentage = pred[i] * 100
        bar_width = int(percentage)
        confidence_bars += f"""
        <div style='margin: 10px 0;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                <span style='font-weight: bold;'>{class_name.upper()}</span>
                <span>{percentage:.1f}%</span>
            </div>
            <div style='background: #e0e0e0; border-radius: 10px; overflow: hidden; height: 20px;'>
                <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); width: {bar_width}%; height: 100%; transition: width 0.3s;'></div>
            </div>
        </div>
        """

    # Create beautiful HTML output
    output = f"""
    <div style='font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto;'>

        <!-- Header -->
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px 15px 0 0; text-align: center;'>
            <h1 style='margin: 0; font-size: 32px;'>🔬 AI Skin Analysis Report</h1>
            <p style='margin: 10px 0 0 0; opacity: 0.9;'>Powered by Deep Learning • MobileNetV2</p>
        </div>

        <!-- Detection Results -->
        <div style='background: white; padding: 30px; border-left: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0;'>
            <h2 style='color: #333; border-bottom: 3px solid #667eea; padding-bottom: 10px; display: inline-block;'>
                {rec['emoji']} Detection Results
            </h2>

            <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                    <div style='flex: 1; min-width: 200px;'>
                        <p style='color: #666; margin: 0;'>Detected Condition</p>
                        <h2 style='color: #667eea; margin: 5px 0; font-size: 28px;'>{rec['title'].upper()}</h2>
                    </div>
                    <div style='flex: 1; min-width: 200px; text-align: right;'>
                        <p style='color: #666; margin: 0;'>Confidence Level</p>
                        <h2 style='color: #764ba2; margin: 5px 0; font-size: 28px;'>{confidence*100:.1f}%</h2>
                    </div>
                </div>
                <p style='color: #888; margin: 15px 0 0 0; font-size: 14px;'>
                    <strong>Status:</strong> {status}
                </p>
            </div>

            <!-- Confidence Breakdown -->
            <h3 style='color: #333; margin-top: 30px;'>📊 Confidence Breakdown</h3>
            <div style='background: #f8f9fa; padding: 20px; border-radius: 10px;'>
                {confidence_bars}
            </div>
        </div>

        <!-- Morning Routine -->
        <div style='background: #fff8dc; padding: 30px; border-left: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0;'>
            <h2 style='color: #333; border-bottom: 3px solid #ffa500; padding-bottom: 10px; display: inline-block;'>
                ☀️ Morning Skincare Routine
            </h2>
            <ol style='line-height: 2; font-size: 16px;'>
                {''.join([f'<li>{step}</li>' for step in rec['morning']])}
            </ol>
        </div>

        <!-- Night Routine -->
        <div style='background: #e6f3ff; padding: 30px; border-left: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0;'>
            <h2 style='color: #333; border-bottom: 3px solid #4a90e2; padding-bottom: 10px; display: inline-block;'>
                🌙 Night Skincare Routine
            </h2>
            <ol style='line-height: 2; font-size: 16px;'>
                {''.join([f'<li>{step}</li>' for step in rec['night']])}
            </ol>
        </div>

        <!-- Product Recommendations -->
        <div style='background: #f0f8f0; padding: 30px; border-left: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0;'>
            <h2 style='color: #333; border-bottom: 3px solid #4caf50; padding-bottom: 10px; display: inline-block;'>
                🛒 Recommended Products
            </h2>
            <ul style='line-height: 2; font-size: 16px;'>
                {''.join([f'<li>✓ {product}</li>' for product in rec['products']])}
            </ul>
        </div>

        <!-- Lifestyle Tips -->
        <div style='background: #fff0f5; padding: 30px; border-left: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0;'>
            <h2 style='color: #333; border-bottom: 3px solid #ff69b4; padding-bottom: 10px; display: inline-block;'>
                🌟 Lifestyle Tips
            </h2>
            <ul style='line-height: 2; font-size: 16px;'>
                {''.join([f'<li>{tip}</li>' for tip in rec['tips']])}
            </ul>
        </div>

        <!-- Timeline & Disclaimer -->
        <div style='background: white; padding: 30px; border: 1px solid #e0e0e0; border-radius: 0 0 15px 15px;'>
            <h3 style='color: #333;'>⏱️ Expected Timeline</h3>
            <ul style='line-height: 1.8;'>
                <li><strong>Week 1-2:</strong> Skin adjusts to new routine</li>
                <li><strong>Week 4-6:</strong> Initial improvements visible</li>
                <li><strong>Month 3:</strong> Significant results</li>
                <li><strong>Month 6+:</strong> Optimal results with consistency</li>
            </ul>

            <div style='background: #fff3cd; border: 2px solid #ffc107; border-radius: 10px; padding: 20px; margin-top: 20px;'>
                <h3 style='color: #856404; margin-top: 0;'>⚠️ Important Disclaimer</h3>
                <p style='color: #856404; margin: 0;'>
                    This is cosmetic guidance based on AI analysis. For persistent concerns, severe conditions,
                    or medical advice, please consult a board-certified dermatologist.
                </p>
            </div>

            <p style='text-align: center; color: #888; margin-top: 20px; font-size: 14px;'>
                💡 <strong>Tip:</strong> Introduce new products one at a time (every 2 weeks) to identify any sensitivities
            </p>
        </div>

    </div>
    """

    return output

# Gradio interface with custom CSS
custom_css = """
#component-0 {
    max-width: 1200px;
    margin: auto;
}
.gradio-container {
    font-family: 'Arial', sans-serif !important;
}
"""

iface = gr.Interface(
    fn=predict_skin,
    inputs=gr.Image(
        sources=["webcam"],
        type="numpy",
        label="📸 Capture Your Face",
        height=450
    ),
    outputs=gr.HTML(label="📋 Analysis Results"),
    title="🌟 AI Skin Detection & Personalized Skincare Recommender",
    description="""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='margin: 0;'>Welcome to AI-Powered Skincare Analysis! 👋</h2>
        <p style='margin: 10px 0 0 0; font-size: 16px;'>Get instant skin analysis and personalized recommendations</p>
    </div>

    <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='margin-top: 0;'>📱 How to Use:</h3>
        <ol style='line-height: 1.8;'>
            <li>Click the <strong>camera icon</strong> to activate your webcam</li>
            <li>Position your <strong>face clearly</strong> in the center</li>
            <li>Ensure <strong>good lighting</strong> for best results</li>
            <li>Click <strong>capture</strong> when ready</li>
            <li>Wait for your personalized analysis!</li>
        </ol>
    </div>

    <div style='background: #e8f5e9; padding: 15px; border-radius: 10px; text-align: center;'>
        <strong>🎯 Detects:</strong> Acne • Dry Skin • Oily Skin • Hyperpigmentation • Normal Skin
    </div>
    """,
    css=custom_css,
    theme=gr.themes.Soft()
)

print("🚀 Launching AI Skin Detection System...")
iface.launch(share=True, debug=True)
print("✅ System ready!")
