'use client';

import ChatInterface from "@/components/chat/Chat";
import ChatTitle from "@/components/chat/Title";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import Sidebar, { ChatPreview } from "@/components/chat/Sidebar";
const SIDEBAR_WIDTH = 250;

export default function Page() {
  const [title, setTitle] = useState("SafeGPT");
  const [selectedId, setSelectedId] = useState<string | undefined>(undefined);
  const [chats, setChats] = useState<ChatPreview[]>([]);

  useEffect(() => {
    setChats(
      Array.from({ length: 101 }, (_, i) => ({
        id: i.toString(),
        title: `Chat ${i}`
      }))
    );
  }, []);

  return <div>
    <div className="flex flex-row h-[70px] items-center justify-start border-b border-gray-200">
      <div style={{width: SIDEBAR_WIDTH, paddingLeft: "20px"}}>
        <Button variant="outline" size="sm" asChild>
          <Link href={`/`} className="flex items-center">
            <ArrowLeft className="mr-1 h-4 w-4" /> Back
          </Link>
        </Button>
      </div>
      <div className="flex-1 flex justify-center">
        <ChatTitle title={title} setTitle={setTitle} />
      </div>
    </div>
    <div className="flex flex-row h-[calc(100vh-70px)]">
      <div style={{width: SIDEBAR_WIDTH}}>
        <Sidebar items={chats} onSelect={setSelectedId} selectedId={selectedId} padding={20} />
      </div>
      <div className="w-[calc(100vw-250px)]">
        <ChatInterface />
      </div>
    </div>
  </div>;
}