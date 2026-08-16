import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

import { ErrorBanner } from "./ErrorBanner";

type Props = { children: ReactNode };
type State = { error: Error | null };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="mx-auto max-w-3xl px-4 py-10">
          <ErrorBanner
            message={this.state.error.message || "The interface failed to render."}
          />
        </div>
      );
    }
    return this.props.children;
  }
}
