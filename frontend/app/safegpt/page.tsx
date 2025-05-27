'use client';

import { useState, Suspense } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { ChevronLeft, ChevronRight, House, FilePlus } from 'lucide-react';
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
import Image from 'next/image';
import { TbLayoutSidebarLeftExpand, TbLayoutSidebarRightExpand } from 'react-icons/tb';

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
      className={`absolute ${
        !collapsed ? 'left-56' : 'left-[90px]'
      } top-[19px] transform -translate-y-1/2 translate-x-1/2 rounded-full border border-gray-200 bg-white hover:bg-gray-50 transition-colors z-20 w-7 h-7 p-0`}
      onClick={onClick}
    >
      {!collapsed ? (
        <ChevronLeft className="h-4 w-4 text-[rgb(85,152,229)]" />
      ) : (
        <ChevronRight className="h-4 w-4 text-[rgb(85,152,229)]" />
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

  const handleSendMessage = async (message: string) => {
    try {
      const newSessionId = await sendMessage(message, apiKey);
      if (newSessionId) {
        router.push(`/safegpt?id=${newSessionId}`);
      }
    } catch (error) {
      if (!(error as Error).message.toLowerCase().includes("api key")) {
        alert("Failed to send message to GPT. Please make sure your API key is correct and you are connected to the internet then try again.");
      }
      throw error;
    }
  };

  const handleUpdateTitle = async (newTitle: string) => {
    try {
      const newSessionId = await updateTitle(newTitle);
      if (newSessionId) {
        router.push(`/safegpt?id=${newSessionId}`);
      }
    } catch (error) {
      alert(error);
    }
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
          className="h-[30px] border-r border-gray-200 transition-all duration-100"
          style={{ width: isSidebarCollapsed ? '0px' : SIDEBAR_WIDTH }}
        />
      </div>
      <div className="flex flex-row h-[40px] items-center justify-start relative bg-white">
        <div
          className={`flex flex-row justify-start px-4 py-4 ${!isSidebarCollapsed && 'border-r'} border-gray-200 transition-all duration-100`}
          style={{ width: SIDEBAR_WIDTH }}
        >
          <Link
            href="/"
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors duration-200"
          >
            <House size={24} className="text-[rgb(85,152,229)]" />
          </Link>

          <button
            onClick={handleNewChat}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors duration-200"
          >
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="text-[rgb(85,152,229)]"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
            >
              <path
                d="M15.6729 3.91287C16.8918 2.69392 18.8682 2.69392 20.0871 3.91287C21.3061 5.13182 21.3061 7.10813 20.0871 8.32708L14.1499 14.2643C13.3849 15.0293 12.3925 15.5255 11.3215 15.6785L9.14142 15.9899C8.82983 16.0344 8.51546 15.9297 8.29289 15.7071C8.07033 15.4845 7.96554 15.1701 8.01005 14.8586L8.32149 12.6785C8.47449 11.6075 8.97072 10.615 9.7357 9.85006L15.6729 3.91287ZM18.6729 5.32708C18.235 4.88918 17.525 4.88918 17.0871 5.32708L11.1499 11.2643C10.6909 11.7233 10.3932 12.3187 10.3014 12.9613L10.1785 13.8215L11.0386 13.6986C11.6812 13.6068 12.2767 13.3091 12.7357 12.8501L18.6729 6.91287C19.1108 6.47497 19.1108 5.76499 18.6729 5.32708ZM11 3.99929C11.0004 4.55157 10.5531 4.99963 10.0008 5.00007C9.00227 5.00084 8.29769 5.00827 7.74651 5.06064C7.20685 5.11191 6.88488 5.20117 6.63803 5.32695C6.07354 5.61457 5.6146 6.07351 5.32698 6.63799C5.19279 6.90135 5.10062 7.24904 5.05118 7.8542C5.00078 8.47105 5 9.26336 5 10.4V13.6C5 14.7366 5.00078 15.5289 5.05118 16.1457C5.10062 16.7509 5.19279 17.0986 5.32698 17.3619C5.6146 17.9264 6.07354 18.3854 6.63803 18.673C6.90138 18.8072 7.24907 18.8993 7.85424 18.9488C8.47108 18.9992 9.26339 19 10.4 19H13.6C14.7366 19 15.5289 18.9992 16.1458 18.9488C16.7509 18.8993 17.0986 18.8072 17.362 18.673C17.9265 18.3854 18.3854 17.9264 18.673 17.3619C18.7988 17.1151 18.8881 16.7931 18.9393 16.2535C18.9917 15.7023 18.9991 14.9977 18.9999 13.9992C19.0003 13.4469 19.4484 12.9995 20.0007 13C20.553 13.0004 21.0003 13.4485 20.9999 14.0007C20.9991 14.9789 20.9932 15.7808 20.9304 16.4426C20.8664 17.116 20.7385 17.7136 20.455 18.2699C19.9757 19.2107 19.2108 19.9756 18.27 20.455C17.6777 20.7568 17.0375 20.8826 16.3086 20.9421C15.6008 21 14.7266 21 13.6428 21H10.3572C9.27339 21 8.39925 21 7.69138 20.9421C6.96253 20.8826 6.32234 20.7568 5.73005 20.455C4.78924 19.9756 4.02433 19.2107 3.54497 18.2699C3.24318 17.6776 3.11737 17.0374 3.05782 16.3086C2.99998 15.6007 2.99999 14.7266 3 13.6428V10.3572C2.99999 9.27337 2.99998 8.39922 3.05782 7.69134C3.11737 6.96249 3.24318 6.3223 3.54497 5.73001C4.02433 4.7892 4.78924 4.0243 5.73005 3.54493C6.28633 3.26149 6.88399 3.13358 7.55735 3.06961C8.21919 3.00673 9.02103 3.00083 9.99922 3.00007C10.5515 2.99964 10.9996 3.447 11 3.99929Z"
                fill="currentColor"
              ></path>
            </svg>
          </button>

          <SidebarToggle
            collapsed={isSidebarCollapsed}
            onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          />
        </div>
        <div className="flex-1 flex justify-center items-center mt-[-16px]">
          <ChatTitle title={title} setTitle={handleUpdateTitle} />
          <div className="absolute right-[20px]">
            <Toggle checked={showRedaction} onChange={handleToggleRedaction} />
          </div>
        </div>
      </div>
      <div className="flex flex-row h-[calc(100vh-70px)] relative">
        <div
          className="transition-all duration-100 overflow-hidden"
          style={{ width: isSidebarCollapsed ? '0px' : SIDEBAR_WIDTH }}
        >
          <Sidebar
            items={previews}
            onSelect={handleSelectChat}
            selectedId={selectedId || undefined}
            onDelete={handleDeleteChat}
            padding={20}
          />
        </div>
        <div
          className="transition-all duration-100"
          style={{ width: isSidebarCollapsed ? '100vw' : 'calc(100vw - 250px)' }}
        >
          <ChatInterface
            messages={messages}
            onSendMessage={handleSendMessage}
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
