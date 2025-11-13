import cv2
import numpy as np

class Detection:
    templates = [cv2.imread(f'src/data/chevron{i}.png', cv2.IMREAD_COLOR) for i in range(1, 4)]
    
    @staticmethod
    def get_chevron_info(img):
        templates = Detection.templates
        if any(template is None for template in templates):
            print("Error: One or more template images not found.")
            return None, None, None
        
        speed_up_colour = templates[0][0, 0]
        results = [cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED) for template in templates]
        
        for result in results:
            print("Max match value:", np.max(result))
        
        threshold = 0.3
        locs = [np.where(result >= threshold) for result in results]
        loc = np.hstack(locs)
        
        # Fixed: Check if arrays are empty properly
        if loc[0].size == 0 or loc[1].size == 0:
            return None, None, None

        position_red = loc[1][0] + templates[0].shape[1] // 2
        position_blue = loc[1][-1] + templates[0].shape[1] // 2

        # Fixed: Correct boundary check - positions are WIDTH (x-coordinates), compare to width
        if position_red >= img.shape[1] or position_blue >= img.shape[1]:
            print("Warning: Detected position is out of image bounds.")
            return None, None, None

        # Fixed: Sample colors from correct locations - need y-coordinates
        # Get y-coordinates from detected locations
        y_red = loc[0][0] + templates[0].shape[0] // 2
        y_blue = loc[0][-1] + templates[0].shape[0] // 2

        # Ensure y-coordinates are also in bounds
        if y_red >= img.shape[0] or y_blue >= img.shape[0]:
            return None, None, None

        colour_red = img[y_red, position_red]
        colour_blue = img[y_blue, position_blue]
        
        is_speed_up_colour = np.all(colour_red == speed_up_colour) or np.all(colour_blue == speed_up_colour)
        
        detected_chevrons = [pt for i, template in enumerate(templates) for pt in zip(*locs[i][::-1])]
        
        # Visualization
        img_copy = img.copy()
        for i, template in enumerate(templates):
            for pt in zip(*locs[i][::-1]):
                cv2.rectangle(img_copy, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0,0,255), 2)
        cv2.imshow('Detected Chevrons', img_copy)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        
        return position_red, position_blue, is_speed_up_colour, detected_chevrons
