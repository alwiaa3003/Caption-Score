// Input validation for POST /api/predict. Mirrors the model service's Pydantic rules so bad
// input is rejected here (400) before any network call.
import { z } from 'zod';

export const MAX_CAPTION_CHARS = 2200;
export const MAX_HASHTAGS = 30;
const HASHTAG = /^#?\w{1,100}$/u;

export const predictSchema = z
  .object({
    caption: z
      .string({ error: 'caption must be a string' })
      .max(MAX_CAPTION_CHARS, `caption must be at most ${MAX_CAPTION_CHARS} characters`)
      .refine((s) => s.trim().length > 0, 'caption must not be empty'),
    hashtags: z
      .array(z.string().trim().regex(HASHTAG, 'hashtags may only contain letters, digits and _'))
      .max(MAX_HASHTAGS, `at most ${MAX_HASHTAGS} hashtags`)
      .optional(),
    post_type: z.enum(['video', 'photo', 'reel', 'other']).optional(),
    duration_sec: z.number().positive().max(3600).optional(),
  })
  .strict(); // unknown fields are rejected, so nothing unexpected gets stored

export const historyQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(100).default(20),
});

/** Express middleware factory: validates req[source] and replaces it with the parsed value. */
export function validate(schema, source = 'body') {
  return (req, res, next) => {
    const result = schema.safeParse(req[source]);
    if (!result.success) {
      return res.status(400).json({
        error: 'Invalid input',
        details: result.error.issues.map((i) => ({
          field: i.path.join('.') || '(body)',
          message: i.message,
        })),
      });
    }
    req.validated = result.data;
    return next();
  };
}
