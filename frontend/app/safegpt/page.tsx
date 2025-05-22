'use client';

import ChatInterface from "@/components/chat/Chat";
import ChatTitle from "@/components/chat/Title";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import Sidebar, { ChatPreview } from "@/components/chat/Sidebar";
import { useRouter, useSearchParams } from "next/navigation";
import useSafeGPT from "./useSafeGPT";

const SIDEBAR_WIDTH = 250;

export default function Page() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedId = searchParams.get('id');
  const { title, updateTitle, previews, messages, sendMessage } = useSafeGPT(selectedId || "new");
  const [showRedaction, setShowRedaction] = useState<boolean>(false);

  const handleToggleRedaction = () => {
    setShowRedaction(prev => !prev);
  }

  const handleSelectChat = (id: string) => {
    router.push(`/safegpt?id=${id}`);
  };
  const handleNewChat = () => {
    router.push(`/safegpt?id=new`);
  };
  console.log("Previews...", previews);

  return <div>
    <div className="flex flex-row h-[70px] items-center justify-start relative bg-white">
      <div className="absolute bottom-0 left-[17%] right-0 h-[1px] bg-gray-200"></div>
      <div style={{ width: SIDEBAR_WIDTH, paddingLeft: "20px" }}>
        <Button variant="outline" size="sm" asChild>
          <Link href={`/`} className="flex items-center">
            <ArrowLeft className="mr-1 h-4 w-4" /> Back
          </Link>
        </Button>
      </div>
      <div className="flex-1 flex justify-center">
        <ChatTitle title={title} setTitle={updateTitle} showRedaction={showRedaction} onToggleRedaction={handleToggleRedaction} />
      </div>
    </div>
    <div className="flex flex-row h-[calc(100vh-70px)]">
      <div style={{ width: SIDEBAR_WIDTH }}>
        <Sidebar
          items={previews}
          onSelect={handleSelectChat}
          selectedId={selectedId || undefined}
          onNewChat={handleNewChat}
          padding={20}
        />
      </div>
      <div className="w-[calc(100vw-250px)]">
        <ChatInterface messages={messages} onSendMessage={sendMessage} />
      </div>
    </div>
  </div>;
}