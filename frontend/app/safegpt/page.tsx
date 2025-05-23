'use client';

import ChatInterface from '@/components/chat/Chat';
import ChatTitle from '@/components/chat/Title';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { useEffect, useState, Suspense } from 'react';
import { Button } from '@/components/ui/button';
import Sidebar, { ChatPreview } from '@/components/chat/Sidebar';
import { useRouter, useSearchParams } from 'next/navigation';
import useSafeGPT from './useSafeGPT';
import Toggle from '@/components/chat/Toggle';
import useApiKeyStore from '@/hooks/useApiKeyStore';

const SIDEBAR_WIDTH = 250;

function SafeGPTContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedId = searchParams.get('id');
  const { title, updateTitle, previews, messages, sendMessage, invalidApiKey, deleteChat } = useSafeGPT(selectedId || 'new');
  const [showRedaction, setShowRedaction] = useState<boolean>(false);
  const { apiKey, saveApiKey } = useApiKeyStore();
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);
  
  const handleToggleRedaction = () => {
    setShowRedaction((prev) => !prev);
  };

  const handleSelectChat = (id: string) => {
    router.push(`/safegpt?id=${id}`);
  };

  const handleNewChat = () => {
    router.push(`/safegpt?id=new`);
  };

  const handleDeleteChat = (id: string) => {
    setChatToDelete(id);
    setIsDeleteDialogOpen(true);
  };

  const handleConfirmDelete = async () => {
    await deleteChat(chatToDelete!);
    setIsDeleteDialogOpen(false);
    setChatToDelete(null);
  };

  const handleCancelDelete = () => {
    setIsDeleteDialogOpen(false);
    setChatToDelete(null);
  };

  return (
    <div>
      <div className="flex flex-row h-[70px] items-center justify-start relative bg-white border-b border-gray-200">
        <div
          className="flex flex-row items-center h-[70px] border-r border-gray-200"
          style={{ width: SIDEBAR_WIDTH, paddingLeft: '20px' }}
        >
          <Button variant="outline" size="sm" asChild>
            <Link href={`/`} className="flex items-center">
              <ArrowLeft className="mr-1 h-4 w-4" /> Back
            </Link>
          </Button>
        </div>
        <div className="flex-1 flex justify-center items-center">
          <ChatTitle title={title} setTitle={updateTitle} />
          <div className="absolute right-[20px]">
            <Toggle checked={showRedaction} onChange={handleToggleRedaction} />
          </div>
        </div>
      </div>
      <div className="flex flex-row h-[calc(100vh-70px)]">
        <div style={{ width: SIDEBAR_WIDTH }}>
          <Sidebar
            items={previews}
            onSelect={handleSelectChat}
            selectedId={selectedId || undefined}
            onNewChat={handleNewChat}
            onDelete={handleDeleteChat}
            padding={20}
          />
        </div>
        <div className="w-[calc(100vw-250px)]">
          <ChatInterface
            messages={messages}
            onSendMessage={(message) => sendMessage(message, apiKey)}
            invalidApiKey={invalidApiKey}
            apiKey={apiKey}
            saveApiKey={saveApiKey}
            showRedaction={showRedaction}
          />
        </div>
      </div>

      {isDeleteDialogOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-medium mb-4">Delete Chat</h3>
            <p className="text-sm text-gray-600 mb-4">
              Are you sure you want to delete this chat? This action cannot be undone.
            </p>
            <div className="flex justify-end space-x-2">
              <Button
                type="button"
                variant="outline"
                onClick={handleCancelDelete}
                style={{
                  color: 'rgb(85,152,229)',
                }}
              >
                Cancel
              </Button>
              <Button
                type="button"
                variant="default"
                onClick={handleConfirmDelete}
                style={{
                  backgroundColor: '#dc2626',
                  color: 'white',
                }}
                onMouseOver={(e) => (e.currentTarget.style.backgroundColor = '#b91c1c')}
                onMouseOut={(e) => (e.currentTarget.style.backgroundColor = '#dc2626')}
              >
                Delete
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <SafeGPTContent />
    </Suspense>
  );
}
