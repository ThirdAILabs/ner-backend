interface Environment {
  allowNewTagsInFeedback: boolean;
  allowCustomTagsInFeedback: boolean;
}

const getEnvironment = (): Environment => {
  return {
    allowNewTagsInFeedback:
      process.env.ALLOW_NEW_TAGS_IN_FEEDBACK?.toLowerCase() === 'true' || false,
    allowCustomTagsInFeedback:
      process.env.ALLOW_CUSTOM_TAGS_IN_FEEDBACK?.toLowerCase() === 'true' || false,
  };
};

export const environment: Environment = getEnvironment();
