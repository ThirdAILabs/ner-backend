import { Feedback } from '@/lib/backend';
import { useEffect, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { nerService } from '@/lib/backend';

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
  console.log("Report Id", reportId);
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

    // Check for duplicate feedback and remove existing entries
    const updatedFeedback = feedback.filter(existingFeedback => {
      const { tag: __, ...existingFeedbackWithoutTag } = existingFeedback.body;

      const isDuplicate =
        feedbackWithoutTag.highlightedText === existingFeedbackWithoutTag.highlightedText &&
        feedbackWithoutTag.objectId === existingFeedbackWithoutTag.objectId &&
        feedbackWithoutTag.startIndex === existingFeedbackWithoutTag.startIndex &&
        feedbackWithoutTag.endIndex === existingFeedbackWithoutTag.endIndex;

      return !isDuplicate;
    });

    setFeedback([...updatedFeedback, { id: uuidv4(), body: newFeedback }]);

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
