import { useState } from "react";
import ScanPage from "./components/ScanPage";
import EnrollmentPage from "./components/EnrollmentPage";

type Mode = "scan" | "enrollment";

function ScanIcon({ active }: { active: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className={`h-5 w-5 ${
        active ? "text-cyan-300" : "text-gray-500"
      }`}
    >
      <path
        d="M4 7V5a1 1 0 0 1 1-1h2M17 4h2a1 1 0 0 1 1 1v2M20 17v2a1 1 0 0 1-1 1h-2M7 20H5a1 1 0 0 1-1-1v-2"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle
        cx="12"
        cy="11"
        r="3"
        stroke="currentColor"
        strokeWidth="2"
      />
      <path
        d="M8 18c.9-2 2.2-3 4-3s3.1 1 4 3"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function UserPlusIcon({ active }: { active: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className={`h-5 w-5 ${
        active ? "text-cyan-300" : "text-gray-500"
      }`}
    >
      <circle
        cx="9"
        cy="8"
        r="3"
        stroke="currentColor"
        strokeWidth="2"
      />
      <path
        d="M3.5 19c.7-3 2.5-4.5 5.5-4.5s4.8 1.5 5.5 4.5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M18 8v6M15 11h6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default function App() {
  const [mode, setMode] = useState<Mode>("scan");
  const [cameraActive, setCameraActive] = useState(false);

  return (
    <div className="min-h-[100dvh] w-full bg-[#02070d] text-gray-100">
      {/* Mobile App Shell */}
      <div className="relative mx-auto flex min-h-[100dvh] w-full max-w-[430px] flex-col overflow-hidden bg-[#06101b] shadow-2xl ring-1 ring-white/10">

        {/* Header */}
        <header className="z-40 shrink-0 border-b border-white/10 bg-[#07131f]/95 px-4 pb-3 pt-[max(0.75rem,env(safe-area-inset-top))] backdrop-blur-xl">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <h1 className="truncate text-base font-black tracking-tight text-white">
                AI Face Vision
              </h1>

              <p className="truncate text-[9px] leading-tight text-gray-500">
                AI Face Recognition System
              </p>
            </div>

            <div className="flex shrink-0 items-center gap-1.5 rounded-full border border-green-400/15 bg-green-400/10 px-2.5 py-1 text-[8px] font-black text-green-300">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-green-400" />
              ONLINE
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main
          className={`
            min-h-0
            flex-1
            overflow-y-auto
            overscroll-contain
            scrollbar-none
            ${cameraActive ? "pb-0" : "pb-[calc(82px+env(safe-area-inset-bottom))]"}
          `}
        >
          {mode === "scan" ? (
            <ScanPage onCameraStateChange={setCameraActive} />
          ) : (
            <EnrollmentPage onCameraStateChange={setCameraActive} />
          )}
        </main>

        {/* =====================================================
            FIXED BOTTOM NAVIGATION
            ===================================================== */}
        {!cameraActive && (
          <nav
            className="
              fixed
              bottom-0
              left-1/2
              z-[9999]
              w-full
              max-w-[430px]
              -translate-x-1/2
              border-t
              border-white/10
              bg-[#07131f]/95
              px-3
              pt-2
              shadow-[0_-12px_35px_rgba(0,0,0,0.45)]
              backdrop-blur-2xl
              pb-[max(0.5rem,env(safe-area-inset-bottom))]
            "
          >
          <div className="grid grid-cols-2 gap-2">

            {/* Scan */}
            <button
              type="button"
              onClick={() => setMode("scan")}
              className={`
                flex
                min-h-14
                touch-manipulation
                flex-col
                items-center
                justify-center
                gap-1
                rounded-2xl
                transition-all
                active:scale-[0.97]
                ${
                  mode === "scan"
                    ? "bg-cyan-400/10 text-cyan-300"
                    : "text-gray-500"
                }
              `}
              aria-label="สแกนใบหน้า"
            >
              <ScanIcon active={mode === "scan"} />

              <span
                className={`text-[10px] font-black ${
                  mode === "scan"
                    ? "text-cyan-300"
                    : "text-gray-500"
                }`}
              >
                สแกนใบหน้า
              </span>

              {mode === "scan" && (
                <span className="h-0.5 w-5 rounded-full bg-cyan-400" />
              )}
            </button>

            {/* Enrollment */}
            <button
              type="button"
              onClick={() => setMode("enrollment")}
              className={`
                flex
                min-h-14
                touch-manipulation
                flex-col
                items-center
                justify-center
                gap-1
                rounded-2xl
                transition-all
                active:scale-[0.97]
                ${
                  mode === "enrollment"
                    ? "bg-cyan-400/10 text-cyan-300"
                    : "text-gray-500"
                }
              `}
              aria-label="ลงทะเบียนใบหน้า"
            >
              <UserPlusIcon active={mode === "enrollment"} />

              <span
                className={`text-[10px] font-black ${
                  mode === "enrollment"
                    ? "text-cyan-300"
                    : "text-gray-500"
                }`}
              >
                ลงทะเบียน
              </span>

              {mode === "enrollment" && (
                <span className="h-0.5 w-5 rounded-full bg-cyan-400" />
              )}
            </button>

          </div>
          </nav>
        )}
      </div>
    </div>
  );
}