import { useToastStore } from '@/stores/toastStore';

class ChatWebSocketService {
  private ws: WebSocket | null = null;
  private messageHandlers: Array<(message: any) => void> = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private url: string = '';
  private hasShownConnectionError = false;

  connect(url: string): void {
    this.url = url;

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log('✅ WebSocket connected');
        this.reconnectAttempts = 0;
        this.hasShownConnectionError = false;

        // Show success toast on reconnection (but not on initial connection)
        if (this.reconnectAttempts > 0) {
          useToastStore.getState().success('toast.websocket.reconnected');
        }
      };

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        // Handle error messages from backend
        if (data.type === 'error') {
          useToastStore.getState().error('toast.websocket.error', { message: data.message });
        }

        this.messageHandlers.forEach(handler => handler(data));
      };

      this.ws.onerror = () => {
        // Show error toast only once per connection attempt
        if (!this.hasShownConnectionError) {
          useToastStore.getState().warning('toast.websocket.connecting');
          this.hasShownConnectionError = true;
        }
      };

      this.ws.onclose = () => {
        if (this.reconnectAttempts === 0) {
          console.log('WebSocket disconnected');
        }
        this.handleReconnect();
      };
    } catch (error) {
      console.warn('Failed to create WebSocket, retrying...');
      this.handleReconnect();
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      console.log(`Reconnecting in ${delay}ms...`);
      setTimeout(() => this.connect(this.url), delay);
    } else {
      // Show error toast if max reconnection attempts reached
      useToastStore.getState().error('toast.websocket.connectionFailed');
    }
  }

  sendMessage(message: string, documentIds?: string[]): void {
    console.log('🚀 Sending message via WebSocket', { message: message.substring(0, 50), documentIds, readyState: this.ws?.readyState });
    if (this.ws?.readyState === WebSocket.OPEN) {
      const payload = {
        type: 'chat_message',
        content: message,
        document_ids: documentIds || []
      };
      console.log('📤 WebSocket payload:', payload);
      this.ws.send(JSON.stringify(payload));
    } else {
      console.error('❌ WebSocket not ready! ReadyState:', this.ws?.readyState);
      useToastStore.getState().error('toast.websocket.notConnected');

      // Try to reconnect
      if (this.url && this.ws?.readyState !== WebSocket.CONNECTING) {
        console.log('🔄 Attempting to reconnect...');
        this.connect(this.url);
      }
    }
  }

  onMessage(handler: (message: any) => void): void {
    this.messageHandlers.push(handler);
  }

  removeMessageHandler(handler: (message: any) => void): void {
    this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
  }

  disconnect(): void {
    this.ws?.close();
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

export const chatWebSocket = new ChatWebSocketService();
