#ifndef ERROR_TYPES_H
#define ERROR_TYPES_H

typedef enum {
    ERROR_X,
    ERROR_Y, 
    ERROR_Z
} error_type_t;

typedef struct {
    size_t x;
    size_t y;
    size_t z;
    error_type_t type;
} error_location_t;

#endif // ERROR_TYPES_H
