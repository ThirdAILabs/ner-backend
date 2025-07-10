interface Environment {
  allowNewTagsInFeedback: boolean;
  allowCustomTagsInFeedback: boolean;
  enterpriseMode: boolean;
}

const getEnvironment = (): Environment => {
  return {
    allowNewTagsInFeedback:
      process.env.ALLOW_NEW_TAGS_IN_FEEDBACK?.toLowerCase() === 'true' || false,
    allowCustomTagsInFeedback:
      process.env.ALLOW_CUSTOM_TAGS_IN_FEEDBACK?.toLowerCase() === 'true' || false,
    enterpriseMode: process.env.NEXT_PUBLIC_ENTERPRISE_MODE?.toLowerCase() === 'true' || false,
  };
};

export const environment: Environment = getEnvironment();
