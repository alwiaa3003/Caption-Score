// One scored caption. No personal data: no IP, user id, or device info is stored.
import mongoose from 'mongoose';

const tokenSchema = new mongoose.Schema({ token: String, weight: Number }, { _id: false });

const suggestionSchema = new mongoose.Schema(
  { hashtag: String, lift: Number, relevance: Number, probability_change: Number },
  { _id: false },
);

const predictionSchema = new mongoose.Schema(
  {
    caption: { type: String, required: true, maxlength: 2200 },
    hashtags: { type: [String], default: [] },
    postType: { type: String, enum: ['video', 'photo', 'reel', 'other', null], default: null },
    durationSec: { type: Number, default: null },
    probability: { type: Number, required: true, min: 0, max: 1 },
    label: { type: String, enum: ['likely', 'unlikely'], required: true },
    modelVersion: { type: String, required: true },
    topPositiveTokens: { type: [tokenSchema], default: [] },
    topNegativeTokens: { type: [tokenSchema], default: [] },
    suggestedHashtags: { type: [suggestionSchema], default: [] },
  },
  { timestamps: { createdAt: true, updatedAt: false }, collection: 'predictions' },
);

predictionSchema.index({ createdAt: -1 });

export const Prediction =
  mongoose.models.Prediction ?? mongoose.model('Prediction', predictionSchema);
