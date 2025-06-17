import { Feedback } from '@/lib/backend';
import { useEffect, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { nerService } from '@/lib/backend';
import toast from 'react-hot-toast';
import { CircleAlert } from 'lucide-react';

export interface DisplayedToken {
  text: string;
  tag: string;
}

export interface FeedbackMetadata {
  highlightedText: string;
  tag: string;
  leftContext: string;
  rightContext: string;
  objectId: string;
  startIndex: number;
  endIndex: number;
}
// TODO: Common function for tokenizing objects

const useFeedbackState = (modelId: string, reportId: string) => {
  const FEEDBACK_STORAGE_KEY = `feedback-${reportId}`;
  const OBJECTS_STORAGE_KEY = `objects-${reportId}`;

  const initialFeedback = JSON.parse(localStorage.getItem(FEEDBACK_STORAGE_KEY) || '[]');
  const [feedback, setFeedback] =
    useState<{ id: string; body: FeedbackMetadata }[]>(initialFeedback);
  const initialObjects = JSON.parse(localStorage.getItem(OBJECTS_STORAGE_KEY) || '{}');
  const [objects, setObjects] = useState<Record<string, Feedback>>(initialObjects);

  // Keep localStorage in sync with feedback state
  useEffect(() => {
    localStorage.setItem(FEEDBACK_STORAGE_KEY, JSON.stringify(feedback));
  }, [feedback, reportId]);

  useEffect(() => {
    localStorage.setItem(OBJECTS_STORAGE_KEY, JSON.stringify(objects));
  }, [objects, reportId]);

  const addFeedback = (
    newFeedback: FeedbackMetadata,
    objectTokens: string[],
    objectTags: string[]
  ) => {
    const { tag: _, ...feedbackWithoutTag } = newFeedback;

    // Track overlapping feedback entries
    const overlappingFeedback: FeedbackMetadata[] = [];

    // Check for duplicate and overlapping feedback
    const updatedFeedback = feedback.filter((existingFeedback) => {
      const { tag: __, ...existingFeedbackWithoutTag } = existingFeedback.body;

      const isDuplicate =
        feedbackWithoutTag.highlightedText === existingFeedbackWithoutTag.highlightedText &&
        feedbackWithoutTag.objectId === existingFeedbackWithoutTag.objectId &&
        feedbackWithoutTag.startIndex === existingFeedbackWithoutTag.startIndex &&
        feedbackWithoutTag.endIndex === existingFeedbackWithoutTag.endIndex;

      const isOverlapped =
        feedbackWithoutTag.objectId === existingFeedbackWithoutTag.objectId &&
        !(
          feedbackWithoutTag.startIndex > existingFeedbackWithoutTag.endIndex ||
          feedbackWithoutTag.endIndex < existingFeedbackWithoutTag.startIndex
        );

      if (isOverlapped && !isDuplicate) {
        overlappingFeedback.push(existingFeedback.body);
      }

      return !isDuplicate;
    });

    if (overlappingFeedback.length > 0) {
      toast.custom(
        (t) => (
          <div className="bg-white rounded-lg shadow-lg border border-red-100 p-4">
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <CircleAlert color="#FF0000" />
                <p className="font-bold">Cannot Add Overlapping Feedback</p>
              </div>
              <p className="text-sm">Your selection overlaps with:</p>
              <ul className="space-y-2 list-none">
                {overlappingFeedback.map((fb, index) => (
                  <li
                    key={index}
                    className="text-sm bg-red-50 p-2 rounded-md border border-red-100"
                  >
                    <div className="flex flex-col gap-1">
                      <span className="font-medium">"{fb.highlightedText}"</span>
                      <div className="flex items-center gap-2 text-xs text-gray-600">
                        <span className="bg-red-100 px-2 py-0.5 rounded">Tag: {fb.tag}</span>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
              <p className="text-sm text-gray-600 italic">
                Please remove existing feedback or select a different range
              </p>
            </div>
          </div>
        ),
        {
          duration: 6000,
          position: 'top-right',
        }
      );
      return; // Don't add new feedback if there are overlaps
    }

    // Add new feedback if no overlaps found
    setFeedback([...updatedFeedback, { id: uuidv4(), body: newFeedback }]);

    // Update objects if needed
    if (!objects[newFeedback.objectId]) {
      setObjects({
        ...objects,
        [newFeedback.objectId]: { tokens: objectTokens, labels: objectTags },
      });
    }
  };

  const submitFeedback = async () => {
    const finetuningSamples: Record<string, Feedback> = {};
    feedback.forEach((feedback) => {
      if (!finetuningSamples[feedback.body.objectId]) {
        finetuningSamples[feedback.body.objectId] = objects[feedback.body.objectId];
      }
      for (let i = feedback.body.startIndex; i <= feedback.body.endIndex; i++) {
        finetuningSamples[feedback.body.objectId].labels[i] = feedback.body.tag;
      }
    });

    for (const objectId in finetuningSamples) {
      try {
        await nerService.submitFeedback(modelId, finetuningSamples[objectId]);
      } catch (error) {
        console.error('Error submitting feedback:', error);
      }
    }

    // Clear temporary feedback and objects
    setFeedback([]);
    setObjects({});
    localStorage.removeItem(FEEDBACK_STORAGE_KEY);
    localStorage.removeItem(OBJECTS_STORAGE_KEY);
  };

  const removeFeedback = (id: string) => {
    setFeedback(feedback.filter((feedback) => feedback.id !== id));
  };

  const displayedFeedback: {
    id: string;
    tokens: DisplayedToken[];
    spotlightStartIndex: number;
    spotlightEndIndex: number;
  }[] = feedback.map((f) => {
    const toWords = (text: string) => {
      return text.split(/\s+/).filter((word) => word.trim() !== '');
    };
    const leftContextWords = toWords(f.body.leftContext);
    const highlightedWords = toWords(f.body.highlightedText);
    const rightContextWords = toWords(f.body.rightContext);
    const spotlightStartIndex = leftContextWords.length;
    const spotlightEndIndex = spotlightStartIndex + highlightedWords.length - 1;
    const tokens = [
      ...leftContextWords.map((word) => ({ text: word, tag: 'O' })),
      ...highlightedWords.map((word) => ({ text: word, tag: f.body.tag })),
      ...rightContextWords.map((word) => ({ text: word, tag: 'O' })),
    ];
    return { id: f.id, tokens, spotlightStartIndex, spotlightEndIndex };
  });

  return {
    displayedFeedback,
    addFeedback,
    removeFeedback,
    submitFeedback,
  };
};

export default useFeedbackState;
