# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the PhotoDB web frontend.

## Architecture Overview

PhotoDB Web is a React Router v7 application that provides a modern, responsive web interface for browsing and viewing photos stored in the PhotoDB system. It's a full-stack TypeScript application with server-side rendering and connects directly to the PostgreSQL database used by the Python processing pipeline.

### Key Technologies

- **React Router v7**: Full-stack React framework with SSR, file-based routing, and TypeScript
- **PostgreSQL**: Direct database integration using `pg` connection pool
- **TailwindCSS**: Utility-first CSS framework with Tailwind v4
- **shadcn/ui**: Component library with Radix UI primitives
- **Vite**: Build tool and development server
- **TypeScript**: Full type safety across frontend and backend

### Application Structure

```
app/
├── components/
│   ├── ui/               # shadcn/ui components (Card, Button, etc.)
│   ├── layout.tsx        # Main layout with navigation
│   ├── breadcrumb.tsx    # Navigation breadcrumbs
│   └── pagination.tsx    # Photo grid pagination
├── lib/
│   ├── db.server.ts      # Database queries and connection pool
│   ├── images.server.ts  # Image serving and metadata utilities
│   ├── settings.ts       # Display settings (confidence thresholds, etc.)
│   └── utils.ts          # Utility functions (cn, etc.)
├── routes/
│   ├── home.tsx          # Home page - years grid view
│   ├── year.tsx          # Year page - months grid view  
│   ├── month.tsx         # Month page - photos grid view
│   ├── photo.tsx         # Individual photo detail view
│   └── api.image.$id.tsx # Image serving API endpoint
├── routes.ts             # Route configuration
├── root.tsx              # Root component with meta tags
└── app.css               # Global Tailwind styles
```

### Database Integration

The web frontend directly queries the same PostgreSQL database used by the Python processing pipeline:

**Key Database Operations (`app/lib/db.server.ts`):**

- `getYearsWithPhotos()`: Get years with photo counts and sample images
- `getMonthsInYear(year)`: Get months in a year with counts and samples
- `getPhotosByMonth(year, month)`: Get paginated photos for a specific month
- `getPhotoDetails(photoId)`: Get full photo details including faces and analysis
- `getPhotoById(photoId)`: Get basic photo info for image serving

**Database Schema Integration:**

- `photo`: Core photo records with normalized paths
- `metadata`: EXIF data, captured dates, GPS coordinates  
- `llm_analysis`: AI-generated descriptions and analysis
- `face`: Detected faces with bounding boxes and person identification

### Route Structure

The application uses hierarchical browsing organized by date:

1. **Home (`/`)**: Grid of years with photo counts and sample images
2. **Year (`/year/:year`)**: Grid of months in the year with samples
3. **Month (`/year/:year/month/:month`)**: Paginated grid of photos in the month
4. **Photo (`/photo/:id`)**: Detailed view of individual photo with metadata
5. **Image API (`/api/image/:id`)**: Serves actual image files with proper MIME types

### UI Components

**Layout System:**

- `Layout`: Main wrapper with navigation header and container
- `Breadcrumb`: Navigation breadcrumbs showing current location

**shadcn/ui Components:**

- `Card`, `CardContent`: Photo and data display containers  
- `Button`: Navigation and action buttons
- `Badge`: Tags and status indicators
- `Checkbox`: Selection interfaces
- `Collapsible`: Expandable content sections

**Custom Components:**

- `Pagination`: Photo grid pagination with page controls

## Development Commands

### Package Management

This project uses `npm` as the package manager:

```bash
npm install                # Install dependencies
npm run dev               # Start development server
npm run build             # Build for production  
npm run start             # Start production server
npm run typecheck         # Run TypeScript type checking
```

### Code Quality

```bash
npm run format            # Format code with Biome
npm run check             # Check and fix code issues with Biome  
npm run lint              # Lint code with Biome (strict mode)
```

### Development Server

```bash
npm run dev               # Start dev server at http://localhost:5173
```

The development server includes:

- Hot Module Replacement (HMR)
- TypeScript compilation
- Automatic reloading on file changes
- Server-side rendering in development

### Building and Deployment

```bash
npm run build             # Create production build in build/ directory
npm run start             # Start production server from build/

# Docker deployment
docker build -t photodb-web .
docker run -p 3000:3000 photodb-web
```

**Production Build Structure:**

```
build/
├── client/               # Static assets and client code
│   ├── assets/          # Bundled JS/CSS with content hashes
│   └── favicon.ico      # Static assets
└── server/              # Server-side rendered code
    └── index.js         # Server entry point
```

## Configuration

### Environment Variables

Required environment variables (typically in `../.env` file):

```bash
# Database connection (required)
DATABASE_URL="postgresql://localhost/photodb"

# Optional configuration
NODE_ENV="development"          # Environment mode
PORT="3000"                    # Production server port
```

### Database Connection

The application automatically:

- Loads environment variables from `../.env` (parent directory)
- Creates PostgreSQL connection pool with optimized settings
- Initializes database schema if tables don't exist
- Uses the same database as the Python processing pipeline

### Display Settings

UI display settings are configured in `app/lib/settings.ts`:

```typescript
export const displaySettings = {
  minTagConfidence: 0.7,           // Tags must have >70% confidence to display
  minFaceConfidenceForTags: 0.5,   // Face detection must be ≥50% to show face tags
  minTaxonomyConfidence: 0.7,      // Apple Vision labels must have >70% confidence
};
```

These settings control:

- **Tag Filtering**: Scene and face tags below `minTagConfidence` are hidden
- **Face Tag Visibility**: Face tags are hidden when face detection confidence is below `minFaceConfidenceForTags`
- **Taxonomy Filtering**: Apple Vision scene classification labels below `minTaxonomyConfidence` are hidden

### TypeScript Configuration

**Path Mapping (`tsconfig.json`):**

- `~/*` maps to `./app/*` for clean imports
- Strict TypeScript settings enabled
- React Router type generation included

**shadcn/ui Configuration (`components.json`):**

- New York style variant
- Neutral base color scheme  
- CSS variables for theming
- Lucide icons integration
- Component aliases configured

## Development Patterns

### File-Based Routing

Routes are defined in `app/routes.ts` and correspond to files in `app/routes/`:

```typescript
export default [
  index("routes/home.tsx"),                    // /
  route("year/:year", "routes/year.tsx"),     // /year/2024
  route("year/:year/month/:month", "routes/month.tsx"), // /year/2024/month/12
  route("photo/:id", "routes/photo.tsx"),     // /photo/123
  route("api/image/:id", "routes/api.image.$id.tsx"), // /api/image/123
] satisfies RouteConfig;
```

### Data Loading Pattern

Each route exports a `loader` function for server-side data fetching:

```typescript
export async function loader({ params }: Route.LoaderArgs) {
  const photos = await getPhotosByMonth(
    parseInt(params.year), 
    parseInt(params.month)
  );
  return { photos };
}

export default function MonthPage({ loaderData }: Route.ComponentProps) {
  const { photos } = loaderData;
  // Component implementation
}
```

### Component Patterns

**UI Components:**

- Use shadcn/ui components for consistent styling
- Follow Tailwind utility-first approach
- Implement responsive design with grid layouts

**Database Queries:**

- All database operations in `app/lib/db.server.ts`
- Use prepared statements with parameterized queries
- Handle database errors gracefully with fallbacks

### Image Serving

Images are served through the `/api/image/:id` endpoint:

```typescript
// Serves images directly from normalized_path with proper MIME types
export async function loader({ params }: Route.LoaderArgs) {
  const photoId = parseInt(params.id);
  const imageBuffer = await getImageBuffer(photoId);
  const photo = await getPhotoById(photoId);
  
  if (!imageBuffer || !photo) {
    throw new Response("Image not found", { status: 404 });
  }

  return new Response(imageBuffer, {
    headers: {
      "Content-Type": getMimeType(photo.normalized_path),
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
}
```

## Testing Strategy

When adding new features or modifying existing functionality:

1. **Type Safety**: Ensure TypeScript compilation passes with `npm run typecheck`
2. **Code Quality**: Run `npm run check` to verify Biome standards
3. **Database Integration**: Test queries against actual PhotoDB database
4. **Responsive Design**: Verify layouts work on mobile and desktop
5. **Image Loading**: Test image serving and error handling
6. **Navigation**: Ensure breadcrumbs and routing work correctly

## Performance Considerations

- **Database Connection Pooling**: Configured for web application load patterns
- **Image Caching**: Long-term caching headers for served images
- **Server-Side Rendering**: Faster initial page loads and SEO benefits
- **Lazy Loading**: Consider implementing for large photo grids
- **Responsive Images**: Could add thumbnail generation for better performance

## Integration with PhotoDB Pipeline

The web frontend integrates seamlessly with the PhotoDB processing pipeline:

- **Shared Database**: Uses the same PostgreSQL database as Python CLI tools
- **File Paths**: Reads `normalized_path` from photos processed by `process-local`  
- **Metadata**: Displays EXIF data extracted by the metadata stage
- **AI Analysis**: Shows LLM descriptions and analysis from `enrich-photos`
- **Face Detection**: Displays detected faces and person identification
