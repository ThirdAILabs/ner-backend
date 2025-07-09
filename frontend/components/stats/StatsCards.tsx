import React from 'react';
import CustomisableCard from '@/components/ui/cards/customisableCard';

// Constants for card dimensions
const CARD_WIDTH = '363px'; // (242 * 1.5)
const CARD_HEIGHT = '264px'; // (150 * 1.65)

//Background image URL
const BACKGROUND_IMAGE_URL = {
  SUCCESS: '/success.png',
  PROGRESS: '/fprogress.svg',
  WAVES: '/waves.png',
  QUOTA: '/quota.png',
  TOKENS: '/token.png',
};

const StatsCards = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
      {/* Completed Scans Card */}
      <CustomisableCard
        className="p-6"
        width={CARD_WIDTH}
        height={CARD_HEIGHT}
        backgroundImage={BACKGROUND_IMAGE_URL.SUCCESS}
      >
        <div className="flex flex-row items-center gap-8">
          <span className="text-[136px] font-bold text-[rgb(101,150,223)]">5</span>
          <div>
            <h3 className="text-2xl font-semibold text-[rgb(102,102,102)]">Completed</h3>
            <h3 className="text-2xl font-semibold text-[rgb(102,102,102)]">Scans</h3>
            <div className="mt-1">
              <p className="text-md text-[rgb(161,161,161)]">Ready to review</p>
            </div>
          </div>
        </div>
      </CustomisableCard>

      {/* Scans In-Progress Card */}
      <CustomisableCard
        className="p-6"
        width={CARD_WIDTH}
        height={CARD_HEIGHT}
        backgroundImage={BACKGROUND_IMAGE_URL.PROGRESS}
      >
        <div className="flex flex-row items-center gap-8">
          <span className="text-[136px] font-bold text-[rgb(101,150,223)]">2</span>
          <div>
            <h3 className="text-2xl font-semibold text-[rgb(102,102,102)]">Scans</h3>
            <h3 className="text-2xl font-semibold text-[rgb(102,102,102)]">In-Progress</h3>
            <div className="mt-1">
              <p className="text-md text-[rgb(161,161,161)]">Sit back and relax.</p>
              <p className="text-md text-[rgb(161,161,161)]">We'll take it from here.</p>
            </div>
          </div>
        </div>
      </CustomisableCard>

      {/* Tokens per second Card */}
      <CustomisableCard
        className="p-6 text-white"
        width={CARD_WIDTH}
        height={CARD_HEIGHT}
        backgroundImage={BACKGROUND_IMAGE_URL.WAVES}
      >
        <div className="flex flex-col mt-[12%] ml-0">
          <span className="text-8xl font-bold">320.23</span>
          <div className="mt-1 ml-3">
            <h3 className="text-2xl font-semibold text-[rgb(192,213,242)]">Tokens per second</h3>
            <p className="text-md opacity-80 text-[rgb(192,213,242)]">
              Throughput is looking good.
            </p>
          </div>
        </div>
      </CustomisableCard>

      {/* Monthly Quota Card */}
      <CustomisableCard
        className="p-2"
        width={CARD_WIDTH}
        height={CARD_HEIGHT}
        backgroundImage={BACKGROUND_IMAGE_URL.QUOTA}
      >
        <div className="flex flex-col">
          <span className="text-8xl font-bold text-white mt-[12%] ml-3">1.01GB</span>
          <div>
            <p className="text-2xl font-semibold text-[rgb(192,213,242)] ml-8">
              / 5GB Monthly Quota
            </p>
            <p className="text-sm opacity-80 text-[rgb(192,213,242)] ml-4">
              Don't worry, you got a lot of room to play with.
            </p>
          </div>
        </div>
      </CustomisableCard>

      {/* Tokens Processed Card */}
      <CustomisableCard
        className="p-6"
        width={CARD_WIDTH}
        height={CARD_HEIGHT}
        backgroundImage={BACKGROUND_IMAGE_URL.TOKENS}
      >
        <div className="flex flex-col mt-[10%] ml-0">
          <span className="text-8xl font-bold text-[rgb(135,228,172)]">325.7K</span>
          <div>
            <h3 className="text-2xl mt-1 font-semibold text-[rgb(135,228,172)]">
              Tokens Processed
            </h3>
          </div>
        </div>
      </CustomisableCard>
    </div>
  );
};

export default StatsCards;
