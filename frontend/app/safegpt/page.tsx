'use client';

import { useState, Suspense } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { ArrowLeft, ChevronLeft, ChevronRight } from 'lucide-react';
import { Box, CircularProgress } from '@mui/material';
import { Typography } from '@mui/material';
import { useHealth } from '@/contexts/HealthProvider';
import useApiKeyStore from '@/hooks/useApiKeyStore';
import { Button } from '@/components/ui/button';
import ChatInterface from '@/components/chat/Chat';
import ChatTitle from '@/components/chat/Title';
import Sidebar from '@/components/chat/Sidebar';
import Toggle from '@/components/chat/Toggle';
import useSafeGPT from '@/hooks/useSafeGPT';

const SIDEBAR_WIDTH = 250;

interface DeleteDialogProps {
  onCancel: () => void;
  onConfirm: () => void;
}

function DeleteDialog({ onCancel, onConfirm }: DeleteDialogProps) {
  return (
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
            onClick={onCancel}
            style={{
              color: 'rgb(85,152,229)',
            }}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="default"
            onClick={onConfirm}
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
  );
}

function SidebarToggle({ collapsed, onClick }: { collapsed: boolean; onClick: () => void }) {
  return (
    <Button
      variant="ghost"
      size="icon"
      className={`absolute ${collapsed ? 'left-10' : 'left-60'} top-[35px] z-10 h-6 w-6 rounded-full border bg-white`}
      onClick={onClick}
    >
      {collapsed ? (
        <ChevronRight className="h-4 w-4" />
      ) : (
        <ChevronLeft className="h-4 w-4" />
      )}
    </Button>
  );
}

function SafeGPTContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedId = searchParams.get('id');

  const { apiKey, saveApiKey } = useApiKeyStore();
  const { title, updateTitle, previews, messages, sendMessage, invalidApiKey, deleteChat } =
    useSafeGPT(selectedId || 'new');

  const [showRedaction, setShowRedaction] = useState<boolean>(false);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState<boolean>(false);

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
      <div className="flex h-0 items-end">
        <div
          className="h-[20px] border-r border-gray-200 transition-all duration-300"
          style={{ width: isSidebarCollapsed ? '0px' : SIDEBAR_WIDTH }}
        />
      </div>
      <div className="flex flex-row h-[70px] items-center justify-start relative bg-white border-b border-gray-200">
        <div
          className="flex flex-row items-center h-[70px] p-4 border-r border-gray-200 transition-all duration-300"
          style={{ width: isSidebarCollapsed ? '0px' : SIDEBAR_WIDTH }}
        >
          <Button variant="outline" size="sm" asChild className={isSidebarCollapsed ? 'hidden' : ''}>
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
      <div className="flex flex-row h-[calc(100vh-90px)] relative">
        <div
          className="transition-all duration-300 overflow-hidden"
          style={{ width: isSidebarCollapsed ? '0px' : SIDEBAR_WIDTH }}
        >
          <Sidebar
            items={previews}
            onSelect={handleSelectChat}
            selectedId={selectedId || undefined}
            onNewChat={handleNewChat}
            onDelete={handleDeleteChat}
            padding={20}
          />
        </div>
        <SidebarToggle
          collapsed={isSidebarCollapsed}
          onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        />
        <div
          className="transition-all duration-300"
          style={{ width: isSidebarCollapsed ? '100vw' : 'calc(100vw - 250px)' }}
        >
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
        <DeleteDialog onCancel={handleCancelDelete} onConfirm={handleConfirmDelete} />
      )}
    </div>
  );
}

function Loading() {
  return (
    <Box
      display="flex"
      flexDirection="column"
      justifyContent="center"
      alignItems="center"
      minHeight="100vh"
      gap={4}
    >
      <CircularProgress sx={{ color: 'rgb(85,152,229)' }} />
      <Typography className="text-gray-500" variant="h5" sx={{ fontWeight: 600, mb: 2 }}>
        Securing the environment
      </Typography>
    </Box>
  );
}

export default function Page() {
  const { healthStatus } = useHealth();
  if (!healthStatus) {
    return <Loading />;
  }

  return (
    <Suspense fallback={<Loading />}>
      <SafeGPTContent />
    </Suspense>
  );
}
