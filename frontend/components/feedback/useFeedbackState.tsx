import { Feedback } from "@/lib/backend";
import { useEffect, useState } from "react";
import { v4 as uuidv4 } from 'uuid';
import { nerService } from "@/lib/backend";

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
  const [feedback, setFeedback] = useState<{ id: string; body: FeedbackMetadata }[]>(initialFeedback);
  const initialObjects = JSON.parse(localStorage.getItem(OBJECTS_STORAGE_KEY) || '{}');
  const [objects, setObjects] = useState<Record<string, Feedback>>(initialObjects);

  // Keep localStorage in sync with feedback state
  useEffect(() => {
    localStorage.setItem(FEEDBACK_STORAGE_KEY, JSON.stringify(feedback));
  }, [feedback, reportId]);
  
  useEffect(() => {
    localStorage.setItem(OBJECTS_STORAGE_KEY, JSON.stringify(objects));
  }, [objects, reportId]);


  const addFeedback = (newFeedback: FeedbackMetadata, objectTokens: string[], objectTags: string[]) => {
    setFeedback([...feedback, { id: uuidv4(), body: newFeedback }]);
    if (!objects[newFeedback.objectId]) {
      setObjects({ ...objects, [newFeedback.objectId]: { tokens: objectTokens, labels: objectTags } });
    }
  }

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
  }

  const removeFeedback = (id: string) => {
    setFeedback(feedback.filter((feedback) => feedback.id !== id));
  }

  const displayedFeedback: { id: string; tokens: DisplayedToken[] }[] = feedback.map((f) => {
    const tokens = [
      { text: f.body.leftContext, tag: 'O' },
      { text: f.body.highlightedText, tag: f.body.tag },
      { text: f.body.rightContext, tag: 'O' }
    ].flatMap((token) => { 
      const {text, tag} = token;
      return text.split(/\s+/).filter((word) => word.trim() !== '').map((word) => ({ text: word, tag }));
    })
    return { id: f.id, tokens };
  });

  return {
    displayedFeedback,
    addFeedback,
    removeFeedback,
    submitFeedback
  }
};

export default useFeedbackState;
